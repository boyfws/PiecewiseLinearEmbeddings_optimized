import pathlib
import sys
from typing import Literal

sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))

import pytest
import torch
from torch import Tensor

from rtdl_num_embeddings import PiecewiseLinearEmbeddings
from src.PiecewiseLinearEmbeddings import OptimizedPiecewiseLinearEmbeddings
from tests.utils import BIN_CASE_NAMES, make_bins, sample_features


FORWARD_RTOL = 2e-5
FORWARD_ATOL = 2e-6

GRAD_RTOL = 5e-5
GRAD_ATOL = 5e-6

TEST_DEVICES = (
    ["cpu", "cuda"]
    if torch.cuda.is_available()
    else ["cpu"]
)

D_EMBEDDING_VALUES = (10, 15, 20)
SEED_VALUES = (42, 0, 1)


@torch.no_grad()
def initialize_original_parameters_(
    module: PiecewiseLinearEmbeddings,
    *,
    seed: int,
) -> None:
    torch.manual_seed(seed)

    if module.linear.weight.is_cuda:
        torch.cuda.manual_seed_all(seed)

    module.linear.weight.uniform_(
        0.05,
        0.25,
    )

    if module.linear.bias is not None:
        module.linear.bias.uniform_(
            0.10,
            0.30,
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
    device: str,
    seed: int,
) -> tuple[
    PiecewiseLinearEmbeddings,
    OptimizedPiecewiseLinearEmbeddings,
]:
    """
    Build the original and optimized modules and copy trainable parameters
    directly through the common linear/linear0 API.
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

    # The optimized implementation must expose the same trainable API.
    assert set(dict(optimized.named_parameters())) == set(
        dict(original.named_parameters())
    )

    optimized.linear.load_state_dict(
        original.linear.state_dict(),
        strict=True,
    )

    if original.linear0 is None:
        assert optimized.linear0 is None
    else:
        assert optimized.linear0 is not None

        optimized.linear0.load_state_dict(
            original.linear0.state_dict(),
            strict=True,
        )

    return original, optimized


# ============================================================
# Forward tests
# ============================================================


@pytest.mark.parametrize(
    "device",
    TEST_DEVICES,
)
@pytest.mark.parametrize(
    "case_name",
    BIN_CASE_NAMES,
)
@pytest.mark.parametrize(
    "version",
    ("A", "B"),
)
@pytest.mark.parametrize(
    "activation",
    (False, True),
)
@pytest.mark.parametrize(
    "seed",
    SEED_VALUES,
)
@pytest.mark.parametrize(
    "d_embedding",
    D_EMBEDDING_VALUES,
)
def test_forward_matches_original(
    device: str,
    case_name: str,
    version: Literal["A", "B"],
    activation: bool,
    seed: int,
    d_embedding: int,
) -> None:
    bins = make_bins(case_name)

    original, optimized = build_equivalent_modules(
        bins=bins,
        d_embedding=d_embedding,
        activation=activation,
        version=version,
        device=device,
        seed=seed,
    )

    x = sample_features(
        bins,
        batch_size=128,
        seed=seed + 1000,
    ).to(device)

    with torch.no_grad():
        expected = original(x)
        actual = optimized(x)

    expected_shape = (
        x.shape[0],
        len(bins),
        d_embedding,
    )

    assert tuple(expected.shape) == expected_shape
    assert tuple(actual.shape) == expected_shape

    assert actual.dtype == expected.dtype
    assert actual.device == expected.device

    assert (
        optimized.get_output_shape()
        == original.get_output_shape()
    )

    torch.testing.assert_close(
        actual,
        expected,
        rtol=FORWARD_RTOL,
        atol=FORWARD_ATOL,
    )


# ============================================================
# Backward tests
# ============================================================


@pytest.mark.parametrize(
    "device",
    TEST_DEVICES,
)
@pytest.mark.parametrize(
    "case_name",
    BIN_CASE_NAMES,
)
@pytest.mark.parametrize(
    "version",
    ("A", "B"),
)
@pytest.mark.parametrize(
    "activation",
    (False, True),
)
@pytest.mark.parametrize(
    "seed",
    SEED_VALUES,
)
@pytest.mark.parametrize(
    "d_embedding",
    D_EMBEDDING_VALUES,
)
def test_backward_matches_original(
    device: str,
    case_name: str,
    version: Literal["A", "B"],
    activation: bool,
    seed: int,
    d_embedding: int,
) -> None:
    bins = make_bins(case_name)

    original, optimized = build_equivalent_modules(
        bins=bins,
        d_embedding=d_embedding,
        activation=activation,
        version=version,
        device=device,
        seed=seed,
    )

    x_base = sample_features(
        bins,
        batch_size=128,
        seed=seed + 2000,
    ).to(device)

    x_original = (
        x_base.detach()
        .clone()
        .requires_grad_(True)
    )

    x_optimized = (
        x_base.detach()
        .clone()
        .requires_grad_(True)
    )

    original.zero_grad(
        set_to_none=True,
    )

    optimized.zero_grad(
        set_to_none=True,
    )

    output_original = original(
        x_original
    )

    output_optimized = optimized(
        x_optimized
    )

    torch.manual_seed(seed + 3000)

    if device == "cuda":
        torch.cuda.manual_seed_all(seed + 3000)

    grad_output = torch.randn_like(
        output_original
    )

    torch.autograd.backward(
        output_original,
        grad_output,
    )

    torch.autograd.backward(
        output_optimized,
        grad_output,
    )

    # --------------------------------------------------------
    # Input gradients
    # --------------------------------------------------------

    assert x_original.grad is not None
    assert x_optimized.grad is not None

    torch.testing.assert_close(
        x_optimized.grad,
        x_original.grad,
        rtol=GRAD_RTOL,
        atol=GRAD_ATOL,
    )

    # --------------------------------------------------------
    # PLE parameter gradients
    # --------------------------------------------------------

    assert original.linear.weight.grad is not None
    assert optimized.linear.weight.grad is not None

    torch.testing.assert_close(
        optimized.linear.weight.grad,
        original.linear.weight.grad,
        rtol=GRAD_RTOL,
        atol=GRAD_ATOL,
    )

    # --------------------------------------------------------
    # Version-specific parameter gradients
    # --------------------------------------------------------

    if version == "A":
        assert original.linear.bias is not None
        assert optimized.linear.bias is not None

        assert original.linear.bias.grad is not None
        assert optimized.linear.bias.grad is not None

        torch.testing.assert_close(
            optimized.linear.bias.grad,
            original.linear.bias.grad,
            rtol=GRAD_RTOL,
            atol=GRAD_ATOL,
        )

        assert original.linear0 is None
        assert optimized.linear0 is None

    else:
        assert original.linear.bias is None
        assert optimized.linear.bias is None

        assert original.linear0 is not None
        assert optimized.linear0 is not None

        assert original.linear0.weight.grad is not None
        assert optimized.linear0.weight.grad is not None

        assert original.linear0.bias.grad is not None
        assert optimized.linear0.bias.grad is not None

        torch.testing.assert_close(
            optimized.linear0.weight.grad,
            original.linear0.weight.grad,
            rtol=GRAD_RTOL,
            atol=GRAD_ATOL,
        )

        torch.testing.assert_close(
            optimized.linear0.bias.grad,
            original.linear0.bias.grad,
            rtol=GRAD_RTOL,
            atol=GRAD_ATOL,
        )
        