import gc
import pathlib
import statistics
import sys

sys.path.insert(
    0,
    str(pathlib.Path(__file__).resolve().parent.parent),
)

from dataclasses import dataclass
from typing import Callable, Literal

import pytest
import torch
from torch import Tensor

from rtdl_num_embeddings import PiecewiseLinearEmbeddings

from src.PiecewiseLinearEmbeddings import (
    OptimizedPiecewiseLinearEmbeddings,
)
from tests.utils import (
    convert_old_ple_state_dict,
    sample_features,
)


# ============================================================
# General configuration
# ============================================================

SEED = 42

FORWARD_RTOL = 2e-4
FORWARD_ATOL = 2e-5

WARMUP_ITERATIONS = 20

# Each reported observation is the average time of this many calls.
ITERATIONS_PER_REPEAT = 10

# Median, mean, min, and max are calculated across these observations.
MEASUREMENT_REPEATS = 10

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

    @property
    def old_dense_encoding_mib(self) -> float:
        """
        Estimated size of one float32 [N, F, B] tensor.
        """
        return (
            self.batch_size
            * self.n_features
            * self.n_bins
            * torch.tensor(
                [],
                dtype=torch.float32,
            ).element_size()
            / 1024**2
        )

    @property
    def embedding_output_mib(self) -> float:
        """
        Estimated size of one float32 [N, F, D] tensor.
        """
        return (
            self.batch_size
            * self.n_features
            * self.d_embedding
            * torch.tensor(
                [],
                dtype=torch.float32,
            ).element_size()
            / 1024**2
        )


batch_size = [2_048, 8_192, 20_000]
n_features = [32, 64, 256]
n_bins = [16, 48, 64]
d_embedding = [12, 16, 32]


BENCHMARK_CASES = (
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
class CudaTimingResult:
    """CUDA execution-time statistics."""

    median_ms: float
    mean_ms: float
    min_ms: float
    max_ms: float


@dataclass(frozen=True)
class CudaBenchmarkResult:
    """CUDA timing and incremental allocated-memory statistics."""

    timing: CudaTimingResult
    peak_allocated_bytes: int

    @property
    def peak_allocated_mib(self) -> float:
        return (
            self.peak_allocated_bytes
            / 1024**2
        )


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
        raise ValueError(
            "n_features must be positive"
        )

    if n_bins <= 0:
        raise ValueError(
            "n_bins must be positive"
        )

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

        bins.append(
            feature_edges
        )

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
    """
    Initialize original PLE parameters with deterministic non-zero values.
    """
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
    OptimizedPiecewiseLinearEmbeddings,
]:
    """
    Build functionally equivalent original and optimized modules.
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

    optimized = OptimizedPiecewiseLinearEmbeddings(
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

    converted_state_dict = convert_old_ple_state_dict(
        original.state_dict()
    )

    load_result = optimized.load_state_dict(
        converted_state_dict,
        strict=False,
    )

    if load_result.unexpected_keys:
        raise RuntimeError(
            "Unexpected keys while loading the converted state dict: "
            f"{load_result.unexpected_keys}"
        )

    return original, optimized


# ============================================================
# CUDA compatibility
# ============================================================

def cuda_build_supports_current_device(
    device: torch.device,
) -> bool:
    """
    Check whether the installed PyTorch build contains kernels for the
    current CUDA architecture.
    """
    major, minor = torch.cuda.get_device_capability(
        device
    )

    architecture = f"sm_{major}{minor}"

    return architecture in torch.cuda.get_arch_list()


# ============================================================
# Timing
# ============================================================

def benchmark_cuda_time(
    fn: Callable[[], object],
    *,
    device: torch.device,
) -> CudaTimingResult:
    """
    Measure CUDA execution time.

    Each observation measures multiple consecutive calls and divides the
    elapsed GPU time by the number of calls. This reduces CUDA Event overhead
    for short operations.
    """

    # Initialize CUDA libraries before the actual warmup.
    result = fn()
    del result

    torch.cuda.synchronize(device)

    # Warm up kernels, allocators, cuBLAS, and GPU clocks.
    for _ in range(WARMUP_ITERATIONS):
        result = fn()
        del result

    torch.cuda.synchronize(device)

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
) -> int:
    """
    Measure incremental peak tensor memory allocated by one call.

    Parameters, inputs, and all tensors that already exist before the measured
    call are excluded from the reported value.
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

    result = fn()

    torch.cuda.synchronize(device)

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
    prepare_memory: Callable[[], None] | None = None,
) -> CudaBenchmarkResult:
    """
    Measure CUDA time and incremental peak allocated memory.
    """
    timing = benchmark_cuda_time(
        fn,
        device=device,
    )

    peak_allocated_bytes = measure_incremental_peak_memory(
        fn,
        device=device,
        prepare=prepare_memory,
    )

    return CudaBenchmarkResult(
        timing=timing,
        peak_allocated_bytes=peak_allocated_bytes,
    )


# ============================================================
# Result reporting
# ============================================================

def print_benchmark_comparison(
    *,
    mode: str,
    case: BenchmarkCase,
    version: str,
    activation: bool,
    original_result: CudaBenchmarkResult,
    optimized_result: CudaBenchmarkResult,
) -> None:
    """
    Print one machine-readable benchmark record.
    """
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

    print()

    print(
        "PLE_CUDA_BENCHMARK"
        f" | mode={mode}"
        f" | case={case.name}"
        f" | batch_size={case.batch_size}"
        f" | n_features={case.n_features}"
        f" | n_bins={case.n_bins}"
        f" | d_embedding={case.d_embedding}"
        f" | version={version}"
        f" | activation={activation}"
        f" | old_dense_encoding_mib="
        f"{case.old_dense_encoding_mib:.3f}"
        f" | embedding_output_mib="
        f"{case.embedding_output_mib:.3f}"
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
def test_cuda_ple_benchmark(
    mode: Literal[
        "forward",
        "forward_backward",
    ],
    case: BenchmarkCase,
    version: Literal["A", "B"],
    activation: bool,
) -> None:
    if not torch.cuda.is_available():
        pytest.skip(
            "CUDA is not available"
        )

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

    x = sample_features(
        bins,
        batch_size=case.batch_size,
        seed=SEED + 1,
    ).to(
        device=device,
        dtype=torch.float32,
    )

    # --------------------------------------------------------
    # Correctness
    # --------------------------------------------------------

    original.eval()
    optimized.eval()

    with torch.inference_mode():
        expected = original(x)
        actual = optimized(x)

    torch.testing.assert_close(
        actual,
        expected,
        rtol=FORWARD_RTOL,
        atol=FORWARD_ATOL,
    )

    del expected
    del actual

    torch.cuda.synchronize(device)

    # --------------------------------------------------------
    # Benchmark callables
    # --------------------------------------------------------

    if mode == "forward":
        original.eval()
        optimized.eval()

        @torch.inference_mode()
        def original_fn() -> Tensor:
            return original(x)

        @torch.inference_mode()
        def optimized_fn() -> Tensor:
            return optimized(x)

        original_prepare_memory = None
        optimized_prepare_memory = None

    else:
        original.train()
        optimized.train()

        generator = torch.Generator(
            device=device
        )

        generator.manual_seed(
            SEED + 2
        )

        grad_output = torch.randn(
            case.batch_size,
            case.n_features,
            case.d_embedding,
            generator=generator,
            device=device,
            dtype=torch.float32,
        )

        def clear_original_gradients() -> None:
            original.zero_grad(
                set_to_none=True
            )

        def clear_optimized_gradients() -> None:
            optimized.zero_grad(
                set_to_none=True
            )

        def original_fn() -> None:
            original.zero_grad(
                set_to_none=True
            )

            output = original(x)

            output.backward(
                grad_output
            )

        def optimized_fn() -> None:
            optimized.zero_grad(
                set_to_none=True
            )

            output = optimized(x)

            output.backward(
                grad_output
            )

        original_prepare_memory = (
            clear_original_gradients
        )

        optimized_prepare_memory = (
            clear_optimized_gradients
        )

    # --------------------------------------------------------
    # Run benchmarks
    # --------------------------------------------------------

    original_result = benchmark_cuda(
        original_fn,
        device=device,
        prepare_memory=original_prepare_memory,
    )

    optimized_result = benchmark_cuda(
        optimized_fn,
        device=device,
        prepare_memory=optimized_prepare_memory,
    )

    print_benchmark_comparison(
        mode=mode,
        case=case,
        version=version,
        activation=activation,
        original_result=original_result,
        optimized_result=optimized_result,
    )

    # Explicitly release large benchmark tensors between parameterized cases.
    del original
    del optimized
    del x

    if mode == "forward_backward":
        del grad_output

    gc.collect()
    torch.cuda.empty_cache()