from __future__ import annotations

import torch
from torch import Tensor


_SUPPORTED_OUTPUT_DTYPES = (
    torch.float32,
    torch.float16,
    torch.bfloat16,
)


@torch.library.register_fake(
    "rtdl_num_embeddings_cuda::forward"
)
def _forward_fake(
    x: Tensor,
    bin_edges: Tensor,
    n_bins: Tensor,
    weight: Tensor,
    output_dtype: torch.dtype,
) -> Tensor:
    torch._check(
        x.ndim == 2,
        lambda: "x must have shape [N, F]",
    )
    torch._check(
        bin_edges.ndim == 2,
        lambda: "bin_edges must have shape [F, B + 1]",
    )
    torch._check(
        n_bins.ndim == 1,
        lambda: "n_bins must have shape [F]",
    )
    torch._check(
        weight.ndim == 3,
        lambda: "weight must have shape [F, B, D]",
    )
    torch._check(
        x.shape[1] == weight.shape[0],
        lambda: "x.shape[1] must equal weight.shape[0]",
    )
    torch._check(
        bin_edges.shape[0] == weight.shape[0],
        lambda: "bin_edges.shape[0] must equal weight.shape[0]",
    )
    torch._check(
        bin_edges.shape[1] == weight.shape[1] + 1,
        lambda: "bin_edges.shape[1] must equal weight.shape[1] + 1",
    )
    torch._check(
        n_bins.shape[0] == weight.shape[0],
        lambda: "n_bins.shape[0] must equal weight.shape[0]",
    )
    torch._check(
        x.dtype == torch.float32,
        lambda: "x must be float32",
    )
    torch._check(
        bin_edges.dtype == torch.float32,
        lambda: "bin_edges must be float32",
    )
    torch._check(
        n_bins.dtype == torch.int32,
        lambda: "n_bins must be int32",
    )
    torch._check(
        weight.dtype == torch.float32,
        lambda: "weight must be float32",
    )
    torch._check(
        output_dtype in _SUPPORTED_OUTPUT_DTYPES,
        lambda: "Unsupported output dtype",
    )

    return torch.empty(
        (
            x.shape[0],
            x.shape[1],
            weight.shape[2],
        ),
        dtype=output_dtype,
        device=x.device,
    )


@torch.library.register_fake(
    "rtdl_num_embeddings_cuda::backward"
)
def _backward_fake(
    grad_output: Tensor,
    x: Tensor,
    bin_edges: Tensor,
    n_bins: Tensor,
    weight: Tensor,
    compute_grad_x: bool,
    compute_grad_weight: bool,
) -> tuple[Tensor, Tensor]:
    torch._check(
        grad_output.ndim == 3,
        lambda: "grad_output must have shape [N, F, D]",
    )
    torch._check(
        grad_output.shape[0] == x.shape[0],
        lambda: "grad_output N mismatch",
    )
    torch._check(
        grad_output.shape[1] == x.shape[1],
        lambda: "grad_output F mismatch",
    )
    torch._check(
        grad_output.shape[2] == weight.shape[2],
        lambda: "grad_output D mismatch",
    )
    torch._check(
        grad_output.dtype in _SUPPORTED_OUTPUT_DTYPES,
        lambda: "Unsupported grad_output dtype",
    )

    grad_x = (
        torch.empty_like(x)
        if compute_grad_x
        else torch.empty(
            (0,),
            dtype=x.dtype,
            device=x.device,
        )
    )
    grad_weight = (
        torch.empty_like(weight)
        if compute_grad_weight
        else torch.empty(
            (0,),
            dtype=weight.dtype,
            device=weight.device,
        )
    )
    return grad_x, grad_weight
