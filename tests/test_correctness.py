import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))

from collections.abc import Sequence

import pytest
import torch
from torch import Tensor, nn

from rtdl_num_embeddings import PiecewiseLinearEmbeddings
from src.PiecewiseLinearEmbeddings import OptimizedPiecewiseLinearEmbeddings
from tests.utils.map_state_dict import convert_old_ple_state_dict
from typing import Literal

FORWARD_RTOL = 2e-5
FORWARD_ATOL = 2e-6


GRAD_RTOL = 5e-5
GRAD_ATOL = 5e-6


TEST_DEVICES = ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]


BIN_CASE_NAMES = (
    "equal",
    "ragged",
    "single_bin",
    "all_single_bin",
)


D_EMBEDDING_VALUES = (10, 15, 20)
SEED_VALUES = (42, 0, 1)

def make_bins(
    case_name: str,
    *,
    dtype: torch.dtype = torch.float32,
) -> list[Tensor]:
    if case_name == "equal":
        return [
            torch.tensor(
                [-4.0, -2.0, 0.0, 1.5, 5.0],
                dtype=dtype,
            ),
            torch.tensor(
                [-3.0, -1.0, 0.5, 2.0, 6.0],
                dtype=dtype,
            ),
            torch.tensor(
                [-5.0, -2.5, -0.5, 3.0, 8.0],
                dtype=dtype,
            ),
        ]

    if case_name == "ragged":
        return [
            torch.tensor(
                [-4.0, -2.0, 0.0, 1.5, 5.0],
                dtype=dtype,
            ),
            torch.tensor(
                [-3.0, 0.0, 4.0],
                dtype=dtype,
            ),
            torch.tensor(
                [-6.0, -3.0, -1.0, 0.5, 2.5, 7.0],
                dtype=dtype,
            ),
            torch.tensor(
                [-2.0, 1.0, 5.0, 9.0],
                dtype=dtype,
            ),
        ]

    if case_name == "single_bin":
        return [
            torch.tensor(
                [-4.0, -2.0, 0.0, 1.5, 5.0],
                dtype=dtype,
            ),
            torch.tensor(
                [-1.0, 3.0],
                dtype=dtype,
            ),
            torch.tensor(
                [-5.0, -1.0, 2.0, 6.0],
                dtype=dtype,
            ),
        ]

    if case_name == "all_single_bin":
        return [
            torch.tensor(
                [-3.0, 2.0],
                dtype=dtype,
            ),
            torch.tensor(
                [-1.0, 4.0],
                dtype=dtype,
            ),
            torch.tensor(
                [-5.0, 5.0],
                dtype=dtype,
            ),
        ]

    raise ValueError(
        f"Unknown bin case: {case_name!r}. "
        f"Expected one of {BIN_CASE_NAMES}"
    )




def sample_features(
    bins: Sequence[Tensor],
    *,
    batch_size: int,
    seed: int,
) -> Tensor:
    # 50 % inside bins 25% below first edge 25 % above last edge
    torch.manual_seed(seed)

    assert batch_size % 4 == 0 

    half_size = batch_size // 2
    quarter_size =  batch_size // 4

    inputs = []

    for edges in bins:
        min_val = edges[0].item()
        max_val = edges[-1].item()

        inside = torch.randn(half_size) * (max_val - min_val) + min_val
        below = min_val - torch.rand(quarter_size).abs() * (max_val - min_val)
        above = max_val + torch.rand(quarter_size).abs() * (max_val - min_val)

        input_tensor = torch.cat([inside, below, above])
        inputs.append(input_tensor)

    return torch.stack(inputs, dim=1)



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
    seed: int 
) -> tuple[
    PiecewiseLinearEmbeddings,
    OptimizedPiecewiseLinearEmbeddings,
]:
    """
    Build the original and optimized modules and transfer the represented
    function through state-dict conversion.
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


    return original, optimized


# ============================================================
# Gradient conversion
# ============================================================

def convert_anchor_grad_to_old_weight_layout(
    *,
    anchor_grad: Tensor,
    n_bins_per_feature: Sequence[int],
    old_weight_shape: torch.Size,
) -> Tensor:
    """
    Convert gradients with respect to anchors into gradients with respect
    to the original bin increments.

    For one feature:

        anchor[j] = sum(increment[k] for k <= j)

    Therefore:

        grad_increment[k] = sum(grad_anchor[j] for j >= k)

    The returned tensor follows the original physical layout, where the last
    real bin is always stored at the final max-bin position.
    """
    if anchor_grad.ndim != 2:
        raise ValueError(
            "anchor_grad must have shape "
            "[sum(n_bins_per_feature), d_embedding]"
        )

    n_features, max_n_bins, d_embedding = old_weight_shape

    if len(n_bins_per_feature) != n_features:
        raise ValueError(
            "n_bins_per_feature must contain one value per feature"
        )

    result = torch.zeros(
        old_weight_shape,
        dtype=anchor_grad.dtype,
        device=anchor_grad.device,
    )

    source_offset = 0

    for feature_idx, feature_n_bins in enumerate(
        n_bins_per_feature
    ):
        feature_anchor_grad = anchor_grad[
            source_offset:
            source_offset + feature_n_bins
        ]

        # Reverse cumulative sum:
        # grad_increment[k] = sum_{j >= k} grad_anchor[j]
        logical_increment_grad = torch.flip(
            torch.cumsum(
                torch.flip(
                    feature_anchor_grad,
                    dims=(0,),
                ),
                dim=0,
            ),
            dims=(0,),
        )

        if feature_n_bins == 1:
            result[
                feature_idx,
                -1,
            ] = logical_increment_grad[0]

        elif feature_n_bins == max_n_bins:
            result[
                feature_idx
            ] = logical_increment_grad

        else:
            result[
                feature_idx,
                :feature_n_bins - 1,
            ] = logical_increment_grad[:-1]

            result[
                feature_idx,
                -1,
            ] = logical_increment_grad[-1]

        source_offset += feature_n_bins

    if source_offset != anchor_grad.shape[0]:
        raise ValueError(
            "The anchor gradient size does not match "
            "n_bins_per_feature"
        )

    return result


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
    d_embedding: int
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
def test_backward_matches_original_after_gradient_mapping(
    device: str,
    case_name: str,
    version: Literal["A", "B"],
    activation: bool,
    seed: int,
    d_embedding: int
) -> None:
    bins = make_bins(case_name)

    original, optimized = build_equivalent_modules(
        bins=bins,
        d_embedding=d_embedding,
        activation=activation,
        version=version,
        device=device,
        seed=seed
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
        set_to_none=True
    )

    optimized.zero_grad(
        set_to_none=True
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
    assert optimized.anchors.grad is not None

    n_bins_per_feature = [
        edges.numel() - 1
        for edges in bins
    ]

    mapped_anchor_grad = (
        convert_anchor_grad_to_old_weight_layout(
            anchor_grad=optimized.anchors.grad,
            n_bins_per_feature=n_bins_per_feature,
            old_weight_shape=original.linear.weight.shape,
        )
    )

    torch.testing.assert_close(
        mapped_anchor_grad,
        original.linear.weight.grad,
        rtol=GRAD_RTOL * 10, # mapped_anchor_grad is a sum of original.linear.weight.grad, so we need to relax the tolerance
        atol=GRAD_ATOL * 10,
    )

    # --------------------------------------------------------
    # Version-specific parameter gradients
    # --------------------------------------------------------

    if version == "A":
        assert original.linear.bias is not None
        assert optimized.bias is not None

        assert original.linear.bias.grad is not None
        assert optimized.bias.grad is not None

        torch.testing.assert_close(
            optimized.bias.grad,
            original.linear.bias.grad,
            rtol=GRAD_RTOL,
            atol=GRAD_ATOL,
        )

    else:
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





