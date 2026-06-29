import gc
import pathlib
import statistics
import sys

sys.path.append(
    str(pathlib.Path(__file__).resolve().parent.parent)
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
    BIN_CASE_NAMES,
    convert_old_ple_state_dict,
    make_bins,
    sample_features,
)


# ============================================================
# Benchmark configuration
# ============================================================

WARMUP_ITERATIONS = 30
MEASUREMENT_ITERATIONS = 100

SEED = 42

FORWARD_RTOL = 2e-5
FORWARD_ATOL = 2e-6


BATCH_SIZES = (
    2_048,
    5_120,
    20_000,
)


D_EMBEDDINGS = (
    8,
    16,
    32,
)


BENCHMARK_VERSIONS: tuple[
    Literal["A", "B"],
    ...,
] = (
    "B",
)


BENCHMARK_ACTIVATIONS = (
    False,
)


BENCHMARK_BIN_CASES = BIN_CASE_NAMES


BENCHMARK_MODES: tuple[
    Literal["forward", "forward_backward"],
    ...,
] = (
    "forward",
    "forward_backward",
)


# ============================================================
# Benchmark result
# ============================================================

@dataclass(frozen=True)
class CudaBenchmarkResult:
    """CUDA benchmark timing and memory statistics."""

    median_ms: float
    mean_ms: float
    min_ms: float
    max_ms: float

    peak_allocated_bytes: int
    peak_reserved_bytes: int

    @property
    def peak_allocated_mib(self) -> float:
        """Return incremental peak allocated memory in MiB."""
        return (
            self.peak_allocated_bytes
            / 1024**2
        )

    @property
    def peak_reserved_mib(self) -> float:
        """Return incremental peak reserved memory in MiB."""
        return (
            self.peak_reserved_bytes
            / 1024**2
        )


# ============================================================
# Module construction
# ============================================================

@torch.no_grad()
def initialize_original_parameters_(
    module: PiecewiseLinearEmbeddings,
    *,
    seed: int,
) -> None:
    """
    Initialize all original PLE parameters with non-zero values.
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

    optimized.load_state_dict(
        converted_state_dict,
        strict=False,
    )

    return original, optimized


# ============================================================
# CUDA compatibility
# ============================================================

def cuda_build_supports_current_device(
    device: torch.device,
) -> bool:
    """
    Check whether the installed PyTorch build contains kernels for
    the current GPU architecture.
    """
    major, minor = torch.cuda.get_device_capability(
        device
    )

    current_architecture = (
        f"sm_{major}{minor}"
    )

    return (
        current_architecture
        in torch.cuda.get_arch_list()
    )


# ============================================================
# CUDA benchmark
# ============================================================

def benchmark_cuda(
    fn: Callable[[], object],
    *,
    device: torch.device,
) -> CudaBenchmarkResult:
    """
    Measure CUDA execution time and incremental peak memory.

    Timing is measured with CUDA events. The reported memory is the
    additional memory allocated above parameters, inputs, and other live
    tensors that existed before the measured function call.
    """

    # --------------------------------------------------------
    # Warmup
    # --------------------------------------------------------

    for _ in range(WARMUP_ITERATIONS):
        result = fn()
        del result

    torch.cuda.synchronize(device)

    # --------------------------------------------------------
    # GPU execution time
    # --------------------------------------------------------

    start_events = [
        torch.cuda.Event(
            enable_timing=True
        )
        for _ in range(MEASUREMENT_ITERATIONS)
    ]

    end_events = [
        torch.cuda.Event(
            enable_timing=True
        )
        for _ in range(MEASUREMENT_ITERATIONS)
    ]

    for start_event, end_event in zip(
        start_events,
        end_events,
        strict=True,
    ):
        start_event.record()

        result = fn()

        end_event.record()

        del result

    torch.cuda.synchronize(device)

    elapsed_times_ms = [
        start_event.elapsed_time(end_event)
        for start_event, end_event in zip(
            start_events,
            end_events,
            strict=True,
        )
    ]

    # --------------------------------------------------------
    # Incremental peak memory
    # --------------------------------------------------------

    gc.collect()

    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)

    baseline_allocated = (
        torch.cuda.memory_allocated(device)
    )

    baseline_reserved = (
        torch.cuda.memory_reserved(device)
    )

    torch.cuda.reset_peak_memory_stats(
        device
    )

    # Keep the result alive until peak memory is recorded.
    result = fn()

    torch.cuda.synchronize(device)

    peak_allocated = (
        torch.cuda.max_memory_allocated(device)
    )

    peak_reserved = (
        torch.cuda.max_memory_reserved(device)
    )

    del result

    incremental_peak_allocated = max(
        0,
        peak_allocated - baseline_allocated,
    )

    incremental_peak_reserved = max(
        0,
        peak_reserved - baseline_reserved,
    )

    return CudaBenchmarkResult(
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
        peak_allocated_bytes=(
            incremental_peak_allocated
        ),
        peak_reserved_bytes=(
            incremental_peak_reserved
        ),
    )


# ============================================================
# Result printing
# ============================================================

def print_benchmark_comparison(
    *,
    mode: str,
    bin_case: str,
    batch_size: int,
    d_embedding: int,
    version: str,
    activation: bool,
    original_result: CudaBenchmarkResult,
    optimized_result: CudaBenchmarkResult,
) -> None:
    """
    Print one machine-readable CUDA benchmark record.
    """
    speedup = (
        original_result.median_ms
        / optimized_result.median_ms
    )

    if optimized_result.peak_allocated_bytes > 0:
        memory_reduction = (
            original_result.peak_allocated_bytes
            / optimized_result.peak_allocated_bytes
        )
    else:
        memory_reduction = float("inf")

    print()

    print(
        "PLE_CUDA_BENCHMARK"
        f" | mode={mode}"
        f" | bins={bin_case}"
        f" | batch_size={batch_size}"
        f" | d_embedding={d_embedding}"
        f" | version={version}"
        f" | activation={activation}"
        f" | original_median_ms="
        f"{original_result.median_ms:.6f}"
        f" | original_mean_ms="
        f"{original_result.mean_ms:.6f}"
        f" | original_min_ms="
        f"{original_result.min_ms:.6f}"
        f" | original_max_ms="
        f"{original_result.max_ms:.6f}"
        f" | optimized_median_ms="
        f"{optimized_result.median_ms:.6f}"
        f" | optimized_mean_ms="
        f"{optimized_result.mean_ms:.6f}"
        f" | optimized_min_ms="
        f"{optimized_result.min_ms:.6f}"
        f" | optimized_max_ms="
        f"{optimized_result.max_ms:.6f}"
        f" | original_peak_allocated_mib="
        f"{original_result.peak_allocated_mib:.3f}"
        f" | optimized_peak_allocated_mib="
        f"{optimized_result.peak_allocated_mib:.3f}"
        f" | original_peak_reserved_mib="
        f"{original_result.peak_reserved_mib:.3f}"
        f" | optimized_peak_reserved_mib="
        f"{optimized_result.peak_reserved_mib:.3f}"
        f" | speedup={speedup:.4f}"
        f" | memory_reduction={memory_reduction:.4f}"
    )


# ============================================================
# Parameterized CUDA benchmark
# ============================================================

@pytest.mark.performance
@pytest.mark.cuda
@pytest.mark.parametrize(
    "mode",
    BENCHMARK_MODES,
    ids=BENCHMARK_MODES,
)
@pytest.mark.parametrize(
    "bin_case",
    BENCHMARK_BIN_CASES,
    ids=BENCHMARK_BIN_CASES,
)
@pytest.mark.parametrize(
    "batch_size",
    BATCH_SIZES,
    ids=lambda value: f"batch={value}",
)
@pytest.mark.parametrize(
    "d_embedding",
    D_EMBEDDINGS,
    ids=lambda value: f"d={value}",
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
    bin_case: str,
    batch_size: int,
    d_embedding: int,
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
        major, minor = (
            torch.cuda.get_device_capability(
                device
            )
        )

        pytest.skip(
            "The installed PyTorch build does not support "
            f"the current CUDA architecture sm_{major}{minor}"
        )

    bins = make_bins(
        bin_case
    )

    original, optimized = build_equivalent_modules(
        bins=bins,
        d_embedding=d_embedding,
        activation=activation,
        version=version,
        device=device,
        seed=SEED,
    )

    x = sample_features(
        bins,
        batch_size=batch_size,
        seed=SEED + 1,
    ).to(
        device=device,
        dtype=torch.float32,
    )

    # --------------------------------------------------------
    # Correctness check
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
    # Build benchmark callables
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
            batch_size,
            len(bins),
            d_embedding,
            generator=generator,
            device=device,
            dtype=torch.float32,
        )

        original_parameters = tuple(
            parameter
            for parameter in original.parameters()
            if parameter.requires_grad
        )

        optimized_parameters = tuple(
            parameter
            for parameter in optimized.parameters()
            if parameter.requires_grad
        )

        def original_fn() -> tuple[Tensor, ...]:
            output = original(x)

            return torch.autograd.grad(
                outputs=output,
                inputs=original_parameters,
                grad_outputs=grad_output,
                create_graph=False,
                retain_graph=False,
            )

        def optimized_fn() -> tuple[Tensor, ...]:
            output = optimized(x)

            return torch.autograd.grad(
                outputs=output,
                inputs=optimized_parameters,
                grad_outputs=grad_output,
                create_graph=False,
                retain_graph=False,
            )

    # --------------------------------------------------------
    # Run benchmarks
    # --------------------------------------------------------

    original_result = benchmark_cuda(
        original_fn,
        device=device,
    )

    optimized_result = benchmark_cuda(
        optimized_fn,
        device=device,
    )

    print_benchmark_comparison(
        mode=mode,
        bin_case=bin_case,
        batch_size=batch_size,
        d_embedding=d_embedding,
        version=version,
        activation=activation,
        original_result=original_result,
        optimized_result=optimized_result,
    )