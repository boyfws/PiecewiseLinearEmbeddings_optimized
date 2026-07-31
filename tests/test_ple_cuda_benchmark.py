import gc
import pathlib
import random
import statistics
import sys
import warnings
import zlib

sys.path.insert(
    0,
    str(pathlib.Path(__file__).resolve().parent.parent),
)

from dataclasses import dataclass
from typing import Callable, Literal

import pytest
import torch
from torch import Tensor, nn

from rtdl_num_embeddings import PiecewiseLinearEmbeddings

from rtdl_num_embeddings_cuda import (
    CudaPiecewiseLinearEmbeddings,
)
from tests.utils import sample_features


# ============================================================
# General configuration
# ============================================================

# This benchmark compares two precision modes:
#
#   float32:
#       - model parameters and floating-point buffers stay float32;
#       - benchmark inputs stay float32;
#       - forward runs without autocast.
#
#   amp_bfloat16:
#       - model parameters and floating-point buffers stay float32;
#       - benchmark inputs stay float32;
#       - forward runs under CUDA autocast(dtype=torch.bfloat16);
#       - backward runs after leaving the autocast context;
#       - GradScaler is intentionally not used for BF16.

SEED = 42

FP32_PRECISION_NAME = "float32"
AMP_DTYPE = torch.bfloat16
AMP_PRECISION_NAME = "amp_bfloat16"

FORWARD_TOLERANCES: dict[
    str,
    tuple[float, float],
] = {
    "float32": (2e-4, 2e-5),
    # The original and optimized implementations use different kernels under
    # AMP. A very small number of values may differ by one BF16 quantization
    # step even though model parameters and inputs remain float32.
    "amp_bfloat16": (5e-2, 1.25e-1),
}

# torch.compile is lazy. These calls happen before timing and are intended to:
#   1. compile forward and, when needed, backward;
#   2. finish Inductor autotuning;
#   3. populate CUDA graph/workspace caches;
#   4. exercise a second batch size for dynamic=True.
COMPILE_STABILIZATION_ITERATIONS = 3

WARMUP_ITERATIONS = 20

# Each reported observation is the average time of this many calls.
ITERATIONS_PER_REPEAT = 10

# Median, mean, min, and max are calculated across these observations.
MEASUREMENT_REPEATS = 10

COMPILE_DYNAMIC = True
COMPILE_FULLGRAPH = True

BENCHMARK_VERSIONS: tuple[
    Literal["A", "B"],
    ...,
] = (
    "B",
)

BENCHMARK_ACTIVATIONS = (
    False,
)

BENCHMARK_MODES: tuple[
    Literal["forward", "forward_backward"],
    ...,
] = (
    "forward",
    "forward_backward",
)

BENCHMARK_PRECISIONS: tuple[
    Literal["float32", "amp_bfloat16"],
    ...,
] = (
    FP32_PRECISION_NAME,
    AMP_PRECISION_NAME,
)


# ============================================================
# Execution configurations
# ============================================================

@dataclass(frozen=True)
class ExecutionConfig:
    """One eager or torch.compile execution configuration."""

    name: str
    compile_mode: Literal[
        "default",
        "max-autotune",
    ] | None

    @property
    def is_compiled(self) -> bool:
        return self.compile_mode is not None


BENCHMARK_EXECUTIONS = (
    ExecutionConfig(
        name="eager",
        compile_mode=None,
    ),
    ExecutionConfig(
        name="compile_default",
        compile_mode="default",
    )
)


class AmpBFloat16Module(nn.Module):
    """Run a float32 module under CUDA AMP with BF16 autocast."""

    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(self, x: Tensor) -> Tensor:
        # Parameters, floating-point buffers, and x remain float32.
        # Autocast chooses BF16 only for eligible CUDA operations.
        with torch.autocast(
            device_type="cuda",
            dtype=AMP_DTYPE,
        ):
            return self.module(x)


def prepare_module_for_precision(
    module: nn.Module,
    *,
    precision_name: Literal["float32", "amp_bfloat16"],
) -> nn.Module:
    """Return the module execution wrapper for the requested precision."""
    if precision_name == FP32_PRECISION_NAME:
        return module

    if precision_name == AMP_PRECISION_NAME:
        return AmpBFloat16Module(module)

    raise ValueError(
        f"Unsupported precision: {precision_name}"
    )


def get_forward_tolerances(
    precision_name: Literal["float32", "amp_bfloat16"],
) -> tuple[float, float]:
    try:
        return FORWARD_TOLERANCES[precision_name]
    except KeyError as error:
        raise ValueError(
            f"Unsupported precision: {precision_name}"
        ) from error


# ============================================================
# Performance cases
# ============================================================

@dataclass(frozen=True)
class BenchmarkCase:
    """One PLE benchmark shape."""

    name: str
    batch_size: int
    n_features: int
    n_bins: int
    d_embedding: int

    def old_dense_encoding_mib(
        self,
        dtype: torch.dtype,
    ) -> float:
        """Estimated size of one [N, F, B] tensor for the tested dtype."""
        return (
            self.batch_size
            * self.n_features
            * self.n_bins
            * torch.empty(
                (),
                dtype=dtype,
            ).element_size()
            / 1024**2
        )

    def embedding_output_mib(
        self,
        dtype: torch.dtype,
    ) -> float:
        """Estimated size of one [N, F, D] tensor for the tested dtype."""
        return (
            self.batch_size
            * self.n_features
            * self.d_embedding
            * torch.empty(
                (),
                dtype=dtype,
            ).element_size()
            / 1024**2
        )


batch_size = [2_048, 8_192, 20_000]
n_features = [32, 64, 256]
n_bins = [16, 48, 64]
d_embedding = [12, 16, 32]


BENCHMARK_CASES = tuple(
    BenchmarkCase(
        name=f"bs={bs}_f={f}_b={b}_d={d}",
        batch_size=bs,
        n_features=f,
        n_bins=b,
        d_embedding=d,
    )
    for bs in batch_size
    for f in n_features
    for b in n_bins
    for d in d_embedding
)


# ============================================================
# Benchmark results
# ============================================================

@dataclass(frozen=True)
class CompileCounterSnapshot:
    """Subset of TorchDynamo global counters used to detect lazy compilation."""

    unique_graphs: int
    calls_captured: int
    frames_ok: int


@dataclass(frozen=True)
class CudaTimingResult:
    """CUDA execution-time statistics."""

    median_ms: float
    mean_ms: float
    min_ms: float
    max_ms: float


@dataclass(frozen=True)
class CudaBenchmarkResult:
    """CUDA timing, memory, and compile-stabilization statistics."""

    timing: CudaTimingResult
    peak_allocated_bytes: int
    stabilization_graphs_created: int
    compile_counter_available: bool

    @property
    def peak_allocated_mib(self) -> float:
        return self.peak_allocated_bytes / 1024**2


# ============================================================
# Benchmark data
# ============================================================


def make_benchmark_bins(
    *,
    n_features: int,
    n_bins: int,
) -> list[Tensor]:
    """
    Create strictly increasing feature-specific bin edges.

    Every feature has the same number of bins, while scale and shift vary
    slightly across features.
    """
    if n_features <= 0:
        raise ValueError("n_features must be positive")

    if n_bins <= 0:
        raise ValueError("n_bins must be positive")

    base_edges = torch.linspace(
        -4.0,
        4.0,
        n_bins + 1,
        dtype=torch.float32,
    )

    bins: list[Tensor] = []

    denominator = max(
        n_features - 1,
        1,
    )

    for feature_idx in range(n_features):
        scale = (
            0.75
            + 0.50
            * feature_idx
            / denominator
        )

        shift = (
            0.08
            * (
                feature_idx % 7
                - 3
            )
        )

        feature_edges = (
            base_edges * scale
            + shift
        )

        bins.append(feature_edges)

    return bins


# ============================================================
# Module initialization
# ============================================================

@torch.no_grad()
def initialize_original_parameters_(
    module: PiecewiseLinearEmbeddings,
    *,
    seed: int,
) -> None:
    """Initialize original PLE parameters deterministically in float32."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    module.linear.weight.uniform_(
        0.05,
        0.25,
    )

    if module.linear.bias is not None:
        module.linear.bias.uniform_(
            0.05,
            0.20,
        )

    if module.linear0 is not None:
        module.linear0.weight.uniform_(
            -0.20,
            0.20,
        )

        module.linear0.bias.uniform_(
            -0.10,
            0.10,
        )


def build_equivalent_modules(
    *,
    bins: list[Tensor],
    d_embedding: int,
    activation: bool,
    version: Literal["A", "B"],
    device: torch.device,
    seed: int,
) -> tuple[
    PiecewiseLinearEmbeddings,
    CudaPiecewiseLinearEmbeddings,
]:
    """
    Build functionally equivalent original and optimized modules.

    The optimized implementation now uses the same trainable-parameter layout
    as the original implementation:

        linear.weight
        linear.bias      # version A only
        linear0.weight   # version B only
        linear0.bias     # version B only

    Parameters, floating-point buffers, and benchmark inputs remain float32.
    The selected precision mode is applied later through an execution wrapper.
    """
    original = PiecewiseLinearEmbeddings(
        bins=bins,
        d_embedding=d_embedding,
        activation=activation,
        version=version,
    ).to(
        device=device,
        dtype=torch.float32,
    )

    optimized = CudaPiecewiseLinearEmbeddings(
        bins=bins,
        d_embedding=d_embedding,
        activation=activation,
        version=version,
    ).to(
        device=device,
        dtype=torch.float32,
    )

    initialize_original_parameters_(
        original,
        seed=seed,
    )

    # The new optimized format matches the original trainable layout,
    # so no state-dict conversion or anchor reconstruction is required.
    optimized.linear.load_state_dict(
        original.linear.state_dict(),
        strict=True,
    )

    if original.linear0 is None:
        if optimized.linear0 is not None:
            raise RuntimeError(
                "The optimized module unexpectedly has linear0 "
                f"for version={version}"
            )
    else:
        if optimized.linear0 is None:
            raise RuntimeError(
                "The optimized module is missing linear0 "
                f"for version={version}"
            )

        optimized.linear0.load_state_dict(
            original.linear0.state_dict(),
            strict=True,
        )

    original_parameter_layout = {
        name: tuple(parameter.shape)
        for name, parameter in original.named_parameters()
    }

    optimized_parameter_layout = {
        name: tuple(parameter.shape)
        for name, parameter in optimized.named_parameters()
    }

    if optimized_parameter_layout != original_parameter_layout:
        raise RuntimeError(
            "Original and optimized trainable-parameter layouts differ: "
            f"original={original_parameter_layout}, "
            f"optimized={optimized_parameter_layout}"
        )

    for module_name, module in (
        ("original", original),
        ("optimized", optimized),
    ):
        for parameter_name, parameter in module.named_parameters():
            if parameter.dtype != torch.float32:
                raise RuntimeError(
                    f"{module_name}.{parameter_name} must remain float32 "
                    f"for AMP, got {parameter.dtype}"
                )

    return original, optimized


# ============================================================
# CUDA and compiler compatibility
# ============================================================


def cuda_build_supports_current_device(
    device: torch.device,
) -> bool:
    """Check whether this PyTorch build contains the current CUDA arch."""
    major, minor = torch.cuda.get_device_capability(
        device
    )

    architecture = f"sm_{major}{minor}"

    return architecture in torch.cuda.get_arch_list()


def reset_compiler_state() -> None:
    """
    Clear in-process Dynamo/Inductor caches between compiled benchmark targets.

    torch.compile caches are associated with Python code objects. Without a
    reset, many parameterized shapes can accumulate cached variants and hit a
    recompile limit, causing a later case to silently fall back to eager.
    """
    compiler_namespace = getattr(
        torch,
        "compiler",
        None,
    )

    compiler_reset = getattr(
        compiler_namespace,
        "reset",
        None,
    )

    if compiler_reset is not None:
        compiler_reset()
    else:
        torch._dynamo.reset()


def get_compile_counter_snapshot() -> CompileCounterSnapshot | None:
    """
    Read TorchDynamo counters used to detect compilation during measurement.

    This is a diagnostic use of an internal API. If a PyTorch release changes
    the counter layout, the benchmark still runs but emits a warning and cannot
    perform the automatic no-recompile assertion. Running with
    TORCH_LOGS=recompiles remains an additional external check.
    """
    try:
        from torch._dynamo.utils import counters

        return CompileCounterSnapshot(
            unique_graphs=int(
                counters["stats"]["unique_graphs"]
            ),
            calls_captured=int(
                counters["stats"]["calls_captured"]
            ),
            frames_ok=int(
                counters["frames"]["ok"]
            ),
        )

    except (AttributeError, KeyError, TypeError):
        return None


def compiled_graph_delta(
    before: CompileCounterSnapshot | None,
    after: CompileCounterSnapshot | None,
) -> int | None:
    if before is None or after is None:
        return None

    return after.unique_graphs - before.unique_graphs


def assert_no_compilation_during_measurement(
    *,
    before: CompileCounterSnapshot | None,
    after: CompileCounterSnapshot | None,
    measurement_name: str,
) -> None:
    """Reject a timing/memory observation contaminated by lazy compilation."""
    graph_delta = compiled_graph_delta(
        before,
        after,
    )

    if graph_delta is None:
        return

    if graph_delta != 0:
        raise RuntimeError(
            "torch.compile created new compiled graphs during "
            f"{measurement_name}: unique_graphs increased by {graph_delta}. "
            "The reported measurement would include an unstable compiled "
            "execution path. Increase compile stabilization or inspect "
            "TORCH_LOGS=recompiles."
        )


# ============================================================
# Benchmark callables and torch.compile stabilization
# ============================================================

@dataclass(frozen=True)
class BenchmarkCallable:
    """Callable and cleanup hooks for one model implementation."""

    fn: Callable[[], object]
    prepare_memory: Callable[[], None] | None


def make_benchmark_callable(
    *,
    module: nn.Module,
    mode: Literal[
        "forward",
        "forward_backward",
    ],
    x: Tensor,
    grad_output: Tensor | None,
) -> BenchmarkCallable:
    if mode == "forward":
        module.eval()

        @torch.inference_mode()
        def fn() -> Tensor:
            return module(x)

        return BenchmarkCallable(
            fn=fn,
            prepare_memory=None,
        )

    module.train()

    if grad_output is None:
        raise ValueError(
            "grad_output is required in forward_backward mode"
        )

    def clear_gradients() -> None:
        module.zero_grad(
            set_to_none=True
        )

    def fn() -> None:
        module.zero_grad(
            set_to_none=True
        )

        output = module(x)

        output.backward(
            grad_output
        )

    return BenchmarkCallable(
        fn=fn,
        prepare_memory=clear_gradients,
    )


def run_module_once(
    *,
    module: nn.Module,
    mode: Literal[
        "forward",
        "forward_backward",
    ],
    x: Tensor,
    grad_output: Tensor | None,
) -> None:
    """Run one synchronized-shape call for compile stabilization."""
    if mode == "forward":
        module.eval()

        with torch.inference_mode():
            output = module(x)

        del output
        return

    module.train()
    module.zero_grad(
        set_to_none=True
    )

    if grad_output is None:
        raise ValueError(
            "grad_output is required in forward_backward mode"
        )

    output = module(x)
    output.backward(grad_output)

    del output


def compile_and_stabilize_module(
    *,
    module: nn.Module,
    execution: ExecutionConfig,
    mode: Literal[
        "forward",
        "forward_backward",
    ],
    x: Tensor,
    grad_output: Tensor | None,
    probe_x: Tensor,
    probe_grad_output: Tensor | None,
    device: torch.device,
) -> tuple[nn.Module, int, bool]:
    """
    Compile outside timing and stabilize forward/backward/autotuning caches.

    The actual benchmark shape is compiled first. A second batch size is then
    executed to exercise dynamic=True. A final set of actual-shape calls leaves
    the benchmark on the exact shape that will be timed.
    """
    if not execution.is_compiled:
        return module, 0, True

    if execution.compile_mode is None:
        raise AssertionError(
            "Compiled execution must define compile_mode"
        )

    reset_compiler_state()

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)

    before = get_compile_counter_snapshot()

    compiled_module = torch.compile(
        module,
        backend="inductor",
        mode=execution.compile_mode,
        dynamic=COMPILE_DYNAMIC,
        fullgraph=COMPILE_FULLGRAPH,
    )

    # First actual-shape call performs lazy compilation and max-autotune.
    run_module_once(
        module=compiled_module,
        mode=mode,
        x=x,
        grad_output=grad_output,
    )
    torch.cuda.synchronize(device)

    # Exercise another batch size. If some operation forces specialization,
    # any resulting recompilation happens here, never inside CUDA timing.
    run_module_once(
        module=compiled_module,
        mode=mode,
        x=probe_x,
        grad_output=probe_grad_output,
    )
    torch.cuda.synchronize(device)

    # Return to the actual benchmark shape and fully stabilize its path.
    for _ in range(COMPILE_STABILIZATION_ITERATIONS):
        run_module_once(
            module=compiled_module,
            mode=mode,
            x=x,
            grad_output=grad_output,
        )

    torch.cuda.synchronize(device)

    if mode == "forward_backward":
        compiled_module.zero_grad(
            set_to_none=True
        )

    after = get_compile_counter_snapshot()

    graph_delta = compiled_graph_delta(
        before,
        after,
    )

    counter_available = graph_delta is not None

    if graph_delta is None:
        warnings.warn(
            "TorchDynamo compile counters are unavailable. "
            "Automatic recompile detection is disabled for this run. "
            "Use TORCH_LOGS=recompiles as an external check.",
            stacklevel=2,
        )
        graph_delta = 0

    return (
        compiled_module,
        graph_delta,
        counter_available,
    )


# ============================================================
# Timing
# ============================================================


def benchmark_cuda_time(
    fn: Callable[[], object],
    *,
    device: torch.device,
    verify_no_compilation: bool,
) -> CudaTimingResult:
    """
    Measure steady-state CUDA execution time.

    All compile/autotune work should already be complete. An additional warmup
    remains here for allocator, clock, and cache stabilization. Compile counters
    are sampled only after warmup and around the actual CUDA Event region.
    """
    # Initialize any libraries not touched by explicit compile stabilization.
    result = fn()
    del result

    torch.cuda.synchronize(device)

    for _ in range(WARMUP_ITERATIONS):
        result = fn()
        del result

    torch.cuda.synchronize(device)

    before_measurement = (
        get_compile_counter_snapshot()
        if verify_no_compilation
        else None
    )

    elapsed_times_ms: list[float] = []

    for _ in range(MEASUREMENT_REPEATS):
        start_event = torch.cuda.Event(
            enable_timing=True
        )

        end_event = torch.cuda.Event(
            enable_timing=True
        )

        start_event.record()

        for _ in range(ITERATIONS_PER_REPEAT):
            result = fn()
            del result

        end_event.record()

        torch.cuda.synchronize(device)

        elapsed_per_call_ms = (
            start_event.elapsed_time(end_event)
            / ITERATIONS_PER_REPEAT
        )

        elapsed_times_ms.append(
            elapsed_per_call_ms
        )

    after_measurement = (
        get_compile_counter_snapshot()
        if verify_no_compilation
        else None
    )

    if verify_no_compilation:
        assert_no_compilation_during_measurement(
            before=before_measurement,
            after=after_measurement,
            measurement_name="CUDA timing",
        )

    return CudaTimingResult(
        median_ms=statistics.median(
            elapsed_times_ms
        ),
        mean_ms=statistics.fmean(
            elapsed_times_ms
        ),
        min_ms=min(
            elapsed_times_ms
        ),
        max_ms=max(
            elapsed_times_ms
        ),
    )


# ============================================================
# Memory
# ============================================================


def measure_incremental_peak_memory(
    fn: Callable[[], object],
    *,
    device: torch.device,
    prepare: Callable[[], None] | None = None,
    verify_no_compilation: bool,
) -> int:
    """
    Measure steady-state incremental peak tensor memory allocated by one call.

    For compiled modes, persistent Inductor/CUDA-graph workspaces created during
    stabilization already exist before the baseline and are intentionally not
    counted. The value represents additional steady-state memory for one call.
    """
    gc.collect()

    if prepare is not None:
        prepare()

    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)

    baseline_allocated = torch.cuda.memory_allocated(
        device
    )

    torch.cuda.reset_peak_memory_stats(
        device
    )

    before_measurement = (
        get_compile_counter_snapshot()
        if verify_no_compilation
        else None
    )

    result = fn()

    torch.cuda.synchronize(device)

    after_measurement = (
        get_compile_counter_snapshot()
        if verify_no_compilation
        else None
    )

    if verify_no_compilation:
        assert_no_compilation_during_measurement(
            before=before_measurement,
            after=after_measurement,
            measurement_name="peak-memory measurement",
        )

    peak_allocated = torch.cuda.max_memory_allocated(
        device
    )

    del result

    return max(
        0,
        peak_allocated - baseline_allocated,
    )


def benchmark_cuda(
    fn: Callable[[], object],
    *,
    device: torch.device,
    prepare_memory: Callable[[], None] | None,
    verify_no_compilation: bool,
    stabilization_graphs_created: int,
    compile_counter_available: bool,
) -> CudaBenchmarkResult:
    """Measure CUDA time and incremental peak allocated memory."""
    timing = benchmark_cuda_time(
        fn,
        device=device,
        verify_no_compilation=verify_no_compilation,
    )

    peak_allocated_bytes = measure_incremental_peak_memory(
        fn,
        device=device,
        prepare=prepare_memory,
        verify_no_compilation=verify_no_compilation,
    )

    return CudaBenchmarkResult(
        timing=timing,
        peak_allocated_bytes=peak_allocated_bytes,
        stabilization_graphs_created=stabilization_graphs_created,
        compile_counter_available=compile_counter_available,
    )


# ============================================================
# Benchmark one implementation
# ============================================================


def benchmark_implementation(
    *,
    module: nn.Module,
    execution: ExecutionConfig,
    mode: Literal[
        "forward",
        "forward_backward",
    ],
    x: Tensor,
    grad_output: Tensor | None,
    probe_x: Tensor,
    probe_grad_output: Tensor | None,
    device: torch.device,
) -> CudaBenchmarkResult:
    benchmark_module: nn.Module = module
    stabilization_graphs_created = 0
    compile_counter_available = True

    if execution.is_compiled:
        (
            benchmark_module,
            stabilization_graphs_created,
            compile_counter_available,
        ) = compile_and_stabilize_module(
            module=module,
            execution=execution,
            mode=mode,
            x=x,
            grad_output=grad_output,
            probe_x=probe_x,
            probe_grad_output=probe_grad_output,
            device=device,
        )

    benchmark_callable = make_benchmark_callable(
        module=benchmark_module,
        mode=mode,
        x=x,
        grad_output=grad_output,
    )

    result = benchmark_cuda(
        benchmark_callable.fn,
        device=device,
        prepare_memory=benchmark_callable.prepare_memory,
        verify_no_compilation=(
            execution.is_compiled
            and compile_counter_available
        ),
        stabilization_graphs_created=(
            stabilization_graphs_created
        ),
        compile_counter_available=(
            compile_counter_available
        ),
    )

    benchmark_module.zero_grad(
        set_to_none=True
    )

    if execution.is_compiled:
        del benchmark_module
        reset_compiler_state()

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)

    return result


# ============================================================
# Measurement order
# ============================================================


def get_measurement_order(
    *,
    mode: str,
    case: BenchmarkCase,
    version: str,
    activation: bool,
    precision_name: str,
    execution: ExecutionConfig,
) -> tuple[
    Literal["original", "optimized"],
    Literal["original", "optimized"],
]:
    """
    Deterministically randomize original/optimized order for every test case.

    This avoids always giving the second implementation a warmer GPU, while
    preserving reproducibility across benchmark runs.
    """
    seed_material = (
        f"{SEED}|{mode}|{case.name}|{version}|{activation}|"
        f"{precision_name}|{execution.name}"
    )

    order_seed = (
        SEED
        + zlib.crc32(
            seed_material.encode("utf-8")
        )
    )

    order: list[
        Literal["original", "optimized"]
    ] = [
        "original",
        "optimized",
    ]

    random.Random(order_seed).shuffle(order)

    return order[0], order[1]


# ============================================================
# Result reporting
# ============================================================


def print_benchmark_comparison(
    *,
    mode: str,
    case: BenchmarkCase,
    version: str,
    activation: bool,
    precision_name: Literal["float32", "amp_bfloat16"],
    execution: ExecutionConfig,
    measurement_order: tuple[str, str],
    original_output_dtype: torch.dtype,
    optimized_output_dtype: torch.dtype,
    original_result: CudaBenchmarkResult,
    optimized_result: CudaBenchmarkResult,
) -> None:
    """Print one machine-readable benchmark record."""
    speedup = (
        original_result.timing.median_ms
        / optimized_result.timing.median_ms
    )

    if original_result.peak_allocated_bytes > 0:
        optimized_to_original_memory_ratio = (
            optimized_result.peak_allocated_bytes
            / original_result.peak_allocated_bytes
        )

        memory_saved_percent = (
            1.0
            - optimized_to_original_memory_ratio
        ) * 100.0

    else:
        optimized_to_original_memory_ratio = float(
            "nan"
        )

        memory_saved_percent = float(
            "nan"
        )

    original_samples_per_second = (
        case.batch_size
        / (
            original_result.timing.median_ms
            / 1_000.0
        )
    )

    optimized_samples_per_second = (
        case.batch_size
        / (
            optimized_result.timing.median_ms
            / 1_000.0
        )
    )

    autocast_dtype_name = (
        "bfloat16"
        if precision_name == AMP_PRECISION_NAME
        else "none"
    )

    print()

    print(
        "PLE_CUDA_BENCHMARK"
        f" | mode={mode}"
        f" | execution={execution.name}"
        f" | compile_mode={execution.compile_mode}"
        f" | compile_dynamic={COMPILE_DYNAMIC}"
        f" | compile_fullgraph={COMPILE_FULLGRAPH}"
        f" | precision={precision_name}"
        f" | parameter_dtype=float32"
        f" | input_dtype=float32"
        f" | autocast_dtype={autocast_dtype_name}"
        f" | original_output_dtype={original_output_dtype}"
        f" | optimized_output_dtype={optimized_output_dtype}"
        f" | measurement_order={measurement_order[0]},{measurement_order[1]}"
        f" | case={case.name}"
        f" | batch_size={case.batch_size}"
        f" | n_features={case.n_features}"
        f" | n_bins={case.n_bins}"
        f" | d_embedding={case.d_embedding}"
        f" | version={version}"
        f" | activation={activation}"
        f" | old_dense_encoding_fp32_mib="
        f"{case.old_dense_encoding_mib(torch.float32):.3f}"
        f" | embedding_output_bfloat16_mib="
        f"{case.embedding_output_mib(torch.bfloat16):.3f}"
        f" | embedding_output_fp32_mib="
        f"{case.embedding_output_mib(torch.float32):.3f}"
        f" | original_median_ms="
        f"{original_result.timing.median_ms:.6f}"
        f" | original_mean_ms="
        f"{original_result.timing.mean_ms:.6f}"
        f" | original_min_ms="
        f"{original_result.timing.min_ms:.6f}"
        f" | original_max_ms="
        f"{original_result.timing.max_ms:.6f}"
        f" | optimized_median_ms="
        f"{optimized_result.timing.median_ms:.6f}"
        f" | optimized_mean_ms="
        f"{optimized_result.timing.mean_ms:.6f}"
        f" | optimized_min_ms="
        f"{optimized_result.timing.min_ms:.6f}"
        f" | optimized_max_ms="
        f"{optimized_result.timing.max_ms:.6f}"
        f" | original_peak_allocated_mib="
        f"{original_result.peak_allocated_mib:.3f}"
        f" | optimized_peak_allocated_mib="
        f"{optimized_result.peak_allocated_mib:.3f}"
        f" | optimized_to_original_memory_ratio="
        f"{optimized_to_original_memory_ratio:.4f}"
        f" | memory_saved_percent="
        f"{memory_saved_percent:.2f}"
        f" | original_samples_per_second="
        f"{original_samples_per_second:.2f}"
        f" | optimized_samples_per_second="
        f"{optimized_samples_per_second:.2f}"
        f" | original_stabilization_graphs_created="
        f"{original_result.stabilization_graphs_created}"
        f" | optimized_stabilization_graphs_created="
        f"{optimized_result.stabilization_graphs_created}"
        f" | original_compile_counter_available="
        f"{original_result.compile_counter_available}"
        f" | optimized_compile_counter_available="
        f"{optimized_result.compile_counter_available}"
        f" | speedup={speedup:.4f}"
    )


# ============================================================
# Parameterized benchmark
# ============================================================

@pytest.mark.performance
@pytest.mark.cuda
@pytest.mark.parametrize(
    "mode",
    BENCHMARK_MODES,
    ids=BENCHMARK_MODES,
)
@pytest.mark.parametrize(
    "case",
    BENCHMARK_CASES,
    ids=lambda case: case.name,
)
@pytest.mark.parametrize(
    "version",
    BENCHMARK_VERSIONS,
    ids=lambda value: f"version={value}",
)
@pytest.mark.parametrize(
    "activation",
    BENCHMARK_ACTIVATIONS,
    ids=lambda value: f"activation={value}",
)
@pytest.mark.parametrize(
    "precision_name",
    BENCHMARK_PRECISIONS,
    ids=lambda value: f"precision={value}",
)
@pytest.mark.parametrize(
    "execution",
    BENCHMARK_EXECUTIONS,
    ids=lambda value: value.name,
)
def test_cuda_ple_benchmark(
    mode: Literal[
        "forward",
        "forward_backward",
    ],
    case: BenchmarkCase,
    version: Literal["A", "B"],
    activation: bool,
    precision_name: Literal["float32", "amp_bfloat16"],
    execution: ExecutionConfig,
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    device = torch.device(
        "cuda",
        torch.cuda.current_device(),
    )

    if not cuda_build_supports_current_device(
        device
    ):
        major, minor = torch.cuda.get_device_capability(
            device
        )

        pytest.skip(
            "The installed PyTorch build does not support "
            f"the current CUDA architecture sm_{major}{minor}"
        )

    if (
        precision_name == AMP_PRECISION_NAME
        and not torch.cuda.is_bf16_supported()
    ):
        pytest.skip(
            "The current CUDA device does not support bfloat16"
        )

    if precision_name not in {
        FP32_PRECISION_NAME,
        AMP_PRECISION_NAME,
    }:
        raise ValueError(
            f"Unsupported precision: {precision_name}"
        )

    bins = make_benchmark_bins(
        n_features=case.n_features,
        n_bins=case.n_bins,
    )

    original, optimized = build_equivalent_modules(
        bins=bins,
        d_embedding=case.d_embedding,
        activation=activation,
        version=version,
        device=device,
        seed=SEED,
    )

    original = prepare_module_for_precision(
        original,
        precision_name=precision_name,
    )

    optimized = prepare_module_for_precision(
        optimized,
        precision_name=precision_name,
    )

    x = sample_features(
        bins,
        batch_size=case.batch_size,
        seed=SEED + 1,
    ).to(
        device=device,
        dtype=torch.float32,
    )

    probe_batch_size = max(
        1,
        case.batch_size // 2,
    )

    probe_x = sample_features(
        bins,
        batch_size=probe_batch_size,
        seed=SEED + 3,
    ).to(
        device=device,
        dtype=torch.float32,
    )

    # --------------------------------------------------------
    # Eager correctness before any torch.compile wrapping
    # --------------------------------------------------------

    original.eval()
    optimized.eval()

    with torch.inference_mode():
        expected = original(x)
        actual = optimized(x)

    original_output_dtype = expected.dtype
    optimized_output_dtype = actual.dtype

    forward_rtol, forward_atol = (
        get_forward_tolerances(
            precision_name
        )
    )

    torch.testing.assert_close(
        actual.float(),
        expected.float(),
        rtol=forward_rtol,
        atol=forward_atol,
    )

    del expected
    del actual

    torch.cuda.synchronize(device)

    # --------------------------------------------------------
    # Backward tensors
    # --------------------------------------------------------

    grad_outputs: dict[str, Tensor | None] = {
        "original": None,
        "optimized": None,
    }

    probe_grad_outputs: dict[str, Tensor | None] = {
        "original": None,
        "optimized": None,
    }

    if mode == "forward_backward":
        generator = torch.Generator(
            device=device
        )

        generator.manual_seed(
            SEED + 2
        )

        base_grad_output = torch.randn(
            case.batch_size,
            case.n_features,
            case.d_embedding,
            generator=generator,
            device=device,
            dtype=torch.float32,
        )

        base_probe_grad_output = torch.randn(
            probe_batch_size,
            case.n_features,
            case.d_embedding,
            generator=generator,
            device=device,
            dtype=torch.float32,
        )

        grad_outputs = {
            "original": base_grad_output.to(
                dtype=original_output_dtype,
            ),
            "optimized": base_grad_output.to(
                dtype=optimized_output_dtype,
            ),
        }

        probe_grad_outputs = {
            "original": base_probe_grad_output.to(
                dtype=original_output_dtype,
            ),
            "optimized": base_probe_grad_output.to(
                dtype=optimized_output_dtype,
            ),
        }

        del base_grad_output
        del base_probe_grad_output

    # --------------------------------------------------------
    # Deterministically randomized measurement order
    # --------------------------------------------------------

    measurement_order = get_measurement_order(
        mode=mode,
        case=case,
        version=version,
        activation=activation,
        precision_name=precision_name,
        execution=execution,
    )

    modules: dict[str, nn.Module] = {
        "original": original,
        "optimized": optimized,
    }

    results: dict[str, CudaBenchmarkResult] = {}

    for implementation_name in measurement_order:
        results[implementation_name] = benchmark_implementation(
            module=modules[implementation_name],
            execution=execution,
            mode=mode,
            x=x,
            grad_output=grad_outputs[implementation_name],
            probe_x=probe_x,
            probe_grad_output=probe_grad_outputs[implementation_name],
            device=device,
        )

    original_result = results["original"]
    optimized_result = results["optimized"]

    print_benchmark_comparison(
        mode=mode,
        case=case,
        version=version,
        activation=activation,
        precision_name=precision_name,
        execution=execution,
        measurement_order=measurement_order,
        original_output_dtype=original_output_dtype,
        optimized_output_dtype=optimized_output_dtype,
        original_result=original_result,
        optimized_result=optimized_result,
    )

    # Explicitly release large tensors between parameterized cases.
    original.zero_grad(set_to_none=True)
    optimized.zero_grad(set_to_none=True)

    del original
    del optimized
    del x
    del probe_x

    del grad_outputs
    del probe_grad_outputs

    reset_compiler_state()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)