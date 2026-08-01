from __future__ import annotations

from typing import Any

import torch
from torch import Tensor


def _setup_forward_context(
    ctx: Any,
    inputs: tuple[
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        torch.dtype,
    ],
    output: tuple[Tensor, Tensor],
) -> None:
    (
        x,
        bin_edges,
        inv_bin_widths,
        n_bins,
        weight,
        _output_dtype,
    ) = inputs
    _x_ple, bin_indices = output
    ctx.save_for_backward(
        x,
        bin_edges,
        inv_bin_widths,
        n_bins,
        weight,
        bin_indices,
    )
    ctx.mark_non_differentiable(bin_indices)
    ctx.set_materialize_grads(False)


def _forward_backward(
    ctx: Any,
    grad_output: Tensor | None,
    _grad_bin_indices: Tensor | None,
) -> tuple[
    Tensor | None,
    None,
    None,
    None,
    Tensor | None,
    None,
]:
    if grad_output is None:
        return None, None, None, None, None, None

    (
        x,
        bin_edges,
        inv_bin_widths,
        n_bins,
        weight,
        bin_indices,
    ) = ctx.saved_tensors
    compute_grad_x = bool(ctx.needs_input_grad[0])
    compute_grad_weight = bool(ctx.needs_input_grad[4])

    grad_x, grad_weight = (
        torch.ops.rtdl_num_embeddings_cuda.backward(
            grad_output.contiguous(),
            x,
            bin_edges,
            inv_bin_widths,
            n_bins,
            weight,
            bin_indices,
            compute_grad_x,
            compute_grad_weight,
        )
    )

    return (
        grad_x if compute_grad_x else None,
        None,
        None,
        None,
        grad_weight if compute_grad_weight else None,
        None,
    )


torch.library.register_autograd(
    "rtdl_num_embeddings_cuda::forward",
    _forward_backward,
    setup_context=_setup_forward_context,
)
