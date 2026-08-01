from __future__ import annotations

from typing import Literal

import torch
from rtdl_num_embeddings import PiecewiseLinearEmbeddings
from torch import Tensor


class CudaPiecewiseLinearEmbeddings(
    PiecewiseLinearEmbeddings
):
    """
    Drop-in CUDA implementation of PiecewiseLinearEmbeddings.

    The fused CUDA operator covers the full PLE branch up to ``x_ple``:

    - right-sided bin search;
    - extrapolating position calculation;
    - logical-to-physical parameter mapping;
    - prefix accumulation;
    - interpolation;
    - first-order backward for ``x`` and ``linear.weight``.

    Bias, activation, version-B ``linear0`` and branch addition remain regular
    PyTorch operations. CPU execution falls back to the original module.

    CUDA contract:

    - ``x`` is contiguous float32;
    - parameters remain float32;
    - no autocast produces float32 ``x_ple``;
    - CUDA FP16/BF16 autocast produces low-precision ``x_ple`` with FP32
      accumulation;
    - first-order autograd and ``torch.compile(fullgraph=True)`` are supported;
    - double backward is not implemented.
    """

    def __init__(
        self,
        bins: list[Tensor],
        d_embedding: int,
        *,
        activation: bool,
        version: Literal[None, "A", "B"] = None,
    ) -> None:
        super().__init__(
            bins,
            d_embedding,
            activation=activation,
            version=version,
        )

        n_bins_list = [
            int(edges.numel() - 1)
            for edges in bins
        ]
        max_n_bins = max(n_bins_list)
        n_features = len(bins)
        base_device = bins[0].device

        for feature_idx, edges in enumerate(bins):
            if edges.device != base_device:
                raise ValueError(
                    "All bin tensors must be on the same device. "
                    f"bins[0].device={base_device}, "
                    f"bins[{feature_idx}].device={edges.device}"
                )

        padded_edges = torch.full(
            (
                n_features,
                max_n_bins + 1,
            ),
            torch.inf,
            dtype=torch.float32,
            device=base_device,
        )
        padded_inv_bin_widths = torch.zeros(
            (
                n_features,
                max_n_bins,
            ),
            dtype=torch.float32,
            device=base_device,
        )

        for feature_idx, edges in enumerate(bins):
            edges_float32 = edges.to(
                device=base_device,
                dtype=torch.float32,
            )
            padded_edges[
                feature_idx,
                : edges_float32.numel(),
            ].copy_(edges_float32)
            padded_inv_bin_widths[
                feature_idx,
                : edges_float32.numel() - 1,
            ].copy_(
                torch.reciprocal(
                    edges_float32[1:]
                    - edges_float32[:-1]
                )
            )

        # Non-persistent buffers preserve strict state_dict compatibility with
        # the original PiecewiseLinearEmbeddings.
        self.register_buffer(
            "_cuda_bin_edges",
            padded_edges,
            persistent=False,
        )
        self.register_buffer(
            "_cuda_inv_bin_widths",
            padded_inv_bin_widths,
            persistent=False,
        )
        self.register_buffer(
            "_cuda_n_bins",
            torch.tensor(
                n_bins_list,
                dtype=torch.int32,
                device=base_device,
            ),
            persistent=False,
        )

    @property
    def n_features(self) -> int:
        return int(self.linear.weight.shape[0])

    @property
    def max_n_bins(self) -> int:
        return int(self.linear.weight.shape[1])

    @property
    def d_embedding(self) -> int:
        return int(self.linear.weight.shape[2])

    def _cuda_output_dtype(self) -> torch.dtype:
        if torch.is_autocast_enabled("cuda"):
            dtype = torch.get_autocast_dtype("cuda")
            if dtype not in (
                torch.float16,
                torch.bfloat16,
            ):
                raise RuntimeError(
                    "CUDA autocast dtype must be float16 or bfloat16, "
                    f"got {dtype}"
                )
            return dtype

        return torch.float32

    def _forward_cuda_x_ple(
        self,
        x: Tensor,
    ) -> Tensor:
        if x.dtype != torch.float32:
            raise TypeError(
                "The fused CUDA path requires float32 x. "
                f"Got {x.dtype}."
            )
        if self.linear.weight.dtype != torch.float32:
            raise TypeError(
                "The fused CUDA path requires float32 parameters. "
                f"Got weight dtype {self.linear.weight.dtype}."
            )
        if not x.is_contiguous():
            raise ValueError(
                "The fused CUDA path requires contiguous x."
            )
        if not self._cuda_bin_edges.is_contiguous():
            raise RuntimeError(
                "_cuda_bin_edges must be contiguous"
            )
        if not self._cuda_inv_bin_widths.is_contiguous():
            raise RuntimeError(
                "_cuda_inv_bin_widths must be contiguous"
            )
        if not self._cuda_n_bins.is_contiguous():
            raise RuntimeError(
                "_cuda_n_bins must be contiguous"
            )
        if not self.linear.weight.is_contiguous():
            raise RuntimeError(
                "linear.weight must be contiguous"
            )

        x_ple, _bin_indices = (
            torch.ops.rtdl_num_embeddings_cuda.forward(
                x,
                self._cuda_bin_edges,
                self._cuda_inv_bin_widths,
                self._cuda_n_bins,
                self.linear.weight,
                self._cuda_output_dtype(),
            )
        )
        return x_ple

    def forward(
        self,
        x: Tensor,
    ) -> Tensor:
        if x.ndim != 2:
            raise ValueError(
                "Only inputs with shape [batch_size, n_features] "
                "are supported"
            )
        if x.shape[1] != self.n_features:
            raise ValueError(
                f"Expected {self.n_features} numerical features, "
                f"got {x.shape[1]}"
            )

        if x.device.type != "cuda":
            return super().forward(x)

        if x.device != self._cuda_bin_edges.device:
            raise RuntimeError(
                "x and the module must be on the same CUDA device"
            )

        x_linear = (
            None
            if self.linear0 is None
            else self.linear0(x)
        )

        x_ple = self._forward_cuda_x_ple(x)

        if self.linear.bias is not None:
            x_ple = x_ple + self.linear.bias

        if self.activation is not None:
            x_ple = self.activation(x_ple)

        return (
            x_ple
            if x_linear is None
            else x_linear + x_ple
        )
