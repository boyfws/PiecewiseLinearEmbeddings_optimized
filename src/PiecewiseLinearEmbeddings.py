import warnings
from typing import Literal, Optional

import torch
import torch.nn.functional as F
from rtdl_num_embeddings import (
    _NLinear,
    _check_bins,
)
from torch import Tensor, nn
from torch.nn.parameter import Parameter


class _FeatureLinearEmbedding(nn.Module):
    """
    Independent linear embedding for every numerical feature.

    Input:
        x: [N, F]

    Output:
        y: [N, F, D]
    """

    def __init__(
        self,
        n_features: int,
        d_embedding: int,
    ) -> None:
        super().__init__()

        self.weight = Parameter(
            torch.empty(
                n_features,
                d_embedding,
            )
        )

        self.bias = Parameter(
            torch.empty(
                n_features,
                d_embedding,
            )
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = self.weight.shape[-1] ** -0.5

        nn.init.uniform_(
            self.weight,
            -bound,
            bound,
        )

        nn.init.uniform_(
            self.bias,
            -bound,
            bound,
        )

    def forward(self, x: Tensor) -> Tensor:
        return torch.addcmul(
            self.bias,
            self.weight,
            x.unsqueeze(-1),
        )


class OptimizedPiecewiseLinearEmbeddings(nn.Module):
    """
    Piecewise-linear embeddings based on trainable anchor vectors.

    Each bin is represented by its right anchor. The left anchor of the first
    bin is fixed at zero. For bin j:

        y = left_anchor + t * (right_anchor - left_anchor)

    where:

        left_anchor  = 0, if j == 0
        left_anchor  = anchor[j - 1], otherwise
        right_anchor = anchor[j]

    Anchors are stored in a compact ragged layout:

        anchors.shape == [sum(n_bins_per_feature), d_embedding]

    Input:
        x: [batch_size, n_features]

    Output:
        y: [batch_size, n_features, d_embedding]
    """

    def __init__(
        self,
        bins: list[Tensor],
        d_embedding: int,
        *,
        activation: bool,
        version: Literal[None, "A", "B"] = None,
    ) -> None:
        if d_embedding <= 0:
            raise ValueError(
                "d_embedding must be a positive integer, "
                f"got {d_embedding}"
            )

        _check_bins(bins)

        if version is None:
            warnings.warn(
                'version is not provided, so version="A" is used',
                stacklevel=2,
            )
            version = "A"

        if version not in {"A", "B"}:
            raise ValueError(
                'version must be either "A" or "B"'
            )

        super().__init__()

        self.version = version
        self.n_features = len(bins)
        self.d_embedding = d_embedding

        n_bins_list = [
            edges.numel() - 1
            for edges in bins
        ]

        self._n_bins_list = tuple(
            int(value)
            for value in n_bins_list
        )

        self.max_n_bins = max(n_bins_list)
        self.total_n_bins = sum(n_bins_list)

        base_device = bins[0].device

        # -----------------------------------------------------
        # Padded edges are required by torch.searchsorted.
        #
        # All edges are intentionally stored in float32.
        # -----------------------------------------------------

        padded_edges = torch.full(
            (
                self.n_features,
                self.max_n_bins + 1,
            ),
            torch.inf,
            dtype=torch.float32,
            device=base_device,
        )

        # -----------------------------------------------------
        # Cache right_edge - left_edge for every real bin.
        #
        # The layout matches padded_edges, so the same flattened
        # indices can be reused in forward.
        # -----------------------------------------------------

        padded_bin_widths = torch.ones(
            (
                self.n_features,
                self.max_n_bins + 1,
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
                :edges_float32.numel(),
            ] = edges_float32

            padded_bin_widths[
                feature_idx,
                :edges_float32.numel() - 1,
            ] = (
                edges_float32[1:]
                - edges_float32[:-1]
            )

        self.register_buffer(
            "bin_edges",
            padded_edges,
        )

        self.register_buffer(
            "bin_widths",
            padded_bin_widths,
        )

        # -----------------------------------------------------
        # Number of real bins per feature.
        # int32 is sufficient because the values are very small.
        # -----------------------------------------------------

        self.register_buffer(
            "n_bins",
            torch.tensor(
                n_bins_list,
                dtype=torch.int32,
                device=base_device,
            ),
        )

        # -----------------------------------------------------
        # Offsets into the flattened padded edge tensor.
        #
        # Each feature occupies max_n_bins + 1 positions.
        # -----------------------------------------------------

        edge_stride = self.max_n_bins + 1

        self.register_buffer(
            "edge_offsets",
            (
                torch.arange(
                    self.n_features,
                    dtype=torch.int32,
                    device=base_device,
                )
                * edge_stride
            ),
        )

        # -----------------------------------------------------
        # Offsets into the ragged anchor tensor.
        #
        # Example:
        #
        # n_bins = [3, 1, 4]
        # offsets = [0, 3, 4]
        #
        # Feature 0 uses anchors [0:3]
        # Feature 1 uses anchors [3:4]
        # Feature 2 uses anchors [4:8]
        # -----------------------------------------------------

        # -----------------------------------------------------
        # Offsets into the temporary padded anchor tensor.
        #
        # Unlike the previous ragged representation, every
        # feature occupies max_n_bins positions.
        # -----------------------------------------------------

        self.register_buffer(
            "anchor_offsets",
            (
                torch.arange(
                    self.n_features,
                    dtype=torch.int32,
                    device=base_device,
                )
                * self.max_n_bins
            ),
        )

        # -----------------------------------------------------
        # Map local bin indices to the parameter layout used by
        # the original PiecewiseLinearEmbeddings.
        #
        # Original layout for a feature with n_bins < max_n_bins:
        #
        #   [first bins..., padding..., last bin]
        #
        # The last real bin always uses channel max_n_bins - 1.
        # -----------------------------------------------------

        real_weight_indices = (
            torch.arange(
                self.max_n_bins,
                dtype=torch.int32,
                device=base_device,
            )
            .unsqueeze(0)
            .expand(
                self.n_features,
                -1,
            )
            .clone()
        )

        real_weight_indices.add_(
            (
                torch.arange(
                    self.n_features,
                    dtype=torch.int32,
                    device=base_device,
                )
                * self.max_n_bins
            ).unsqueeze(1)
        )

        for feature_idx, feature_n_bins in enumerate(
            self._n_bins_list
        ):
            real_weight_indices[
                feature_idx,
                feature_n_bins - 1,
            ] = (
                feature_idx * self.max_n_bins
                + self.max_n_bins
                - 1
            )

        self.register_buffer(
            "real_weight_indices",
            real_weight_indices,
        )

        # -----------------------------------------------------
        # Trainable parameters have the same names and shapes
        # as in the original PiecewiseLinearEmbeddings.
        # -----------------------------------------------------

        is_version_b = version == "B"

        self.linear0 = (
            _FeatureLinearEmbedding(
                self.n_features,
                d_embedding,
            ).to(device=base_device)
            if is_version_b
            else None
        )

        self.linear = _NLinear(
            self.n_features,
            self.max_n_bins,
            d_embedding,
            bias=not is_version_b,
        ).to(device=base_device)

        if is_version_b:
            # Same initialization as in the original version B:
            # initially only the ordinary linear component is active.
            nn.init.zeros_(
                self.linear.weight
            )

        self.activation = (
            nn.ReLU()
            if activation
            else None
        )


    def reset_parameters(self) -> None:
        if self.linear0 is not None:
            self.linear0.reset_parameters()

        self.linear.reset_parameters()

        if self.version == "B":
            nn.init.zeros_(
                self.linear.weight
            )

    def get_output_shape(self) -> torch.Size:
        """Return the output shape without the batch dimension."""
        return torch.Size(
            (
                self.n_features,
                self.d_embedding,
            )
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Compute piecewise-linear embeddings without constructing [N, F, B].
        """
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

        if x.device != self.bin_edges.device:
            raise RuntimeError(
                "x and the module must be on the same device"
            )

        batch_size = x.shape[0]

        # -----------------------------------------------------
        # searchsorted searches along the last dimension.
        #
        # x:                 [N, F]
        # values_by_feature: [F, N]
        # bin_edges:         [F, max_n_bins + 1]
        # -----------------------------------------------------

        values_by_feature = (
            x.transpose(0, 1)
            .contiguous()
            #.to(dtype=torch.float32)
        )

        # [F, N], int32
        local_bin_by_feature = torch.searchsorted(
            self.bin_edges,
            values_by_feature,
            side="right",
            out_int32=True,
        )

        local_bin_by_feature.sub_(1)
        local_bin_by_feature.clamp_min_(0)

        # Clamp values above the last edge to the last real bin.
        last_bin_by_feature = (
            self.n_bins.unsqueeze(1) - 1
        )

        local_bin_by_feature = torch.minimum(
            local_bin_by_feature,
            last_bin_by_feature,
        )

        # -----------------------------------------------------
        # Select the left numerical bin edge and the cached
        # bin width.
        #
        # Flattened indexing allows keeping int32 indices.
        # -----------------------------------------------------

        left_edge_indices = (
            local_bin_by_feature
            + self.edge_offsets.unsqueeze(1)
        )

        flat_edge_indices = (
            left_edge_indices.reshape(-1)
        )

        left_edges = torch.index_select(
            self.bin_edges.reshape(-1),
            dim=0,
            index=flat_edge_indices,
        ).view(
            self.n_features,
            batch_size,
        )

        bin_widths = torch.index_select(
            self.bin_widths.reshape(-1),
            dim=0,
            index=flat_edge_indices,
        ).view(
            self.n_features,
            batch_size,
        )

        # t may be below zero or above one for extrapolation.
        position_by_feature = (
            values_by_feature - left_edges
        )

        position_by_feature.div_(
            bin_widths
        )

        # Convert to the batch-major layout used by the output.
        local_bin = (
            local_bin_by_feature
            .transpose(0, 1)
            .contiguous()
        )

        position = (
            position_by_feature
            .transpose(0, 1)
            .contiguous()
        )

        # -----------------------------------------------------
        # Extract independent bin weights in local-bin order.
        #
        # self.linear.weight follows the original padded layout,
        # while real_bin_weights follows simple local-bin order:
        #
        #   [feature, local_bin, embedding]
        # -----------------------------------------------------

        real_bin_weights = torch.index_select(
            self.linear.weight.reshape(
                -1,
                self.d_embedding,
            ),
            dim=0,
            index=self.real_weight_indices.reshape(-1),
        ).view(
            self.n_features,
            self.max_n_bins,
            self.d_embedding,
        )

        # -----------------------------------------------------
        # Precompute cumulative right anchors once per forward.
        #
        # This tensor depends only on the parameters, not on the
        # batch size. FP32 accumulation prevents the BF16 collapse
        # of neighbouring cumulative anchors.
        # -----------------------------------------------------

        right_anchors = torch.cumsum(
            real_bin_weights,
            dim=1,
            #dtype=torch.float32, # Inheritate data type of real_bin_weights
        ).reshape(
            -1,
            self.d_embedding,
        )

        # -----------------------------------------------------
        # Construct anchor indices directly inside the final
        # [batch_size, n_features, 2] tensor.
        #
        # Column 0: left anchor index.
        # Column 1: right anchor index.
        # -----------------------------------------------------

        anchor_indices = torch.empty(
            (
                batch_size,
                self.n_features,
                2,
            ),
            dtype=local_bin.dtype,
            device=local_bin.device,
        )

        left_indices = anchor_indices[..., 0]
        right_indices = anchor_indices[..., 1]

        # Right anchor:
        #
        #     anchor_offsets + local_bin
        right_indices.copy_(local_bin)
        right_indices.add_(self.anchor_offsets)

        # Left anchor:
        #
        #     anchor_offsets + max(local_bin - 1, 0)
        left_indices.copy_(local_bin)
        left_indices.sub_(1)
        left_indices.clamp_min_(0)
        left_indices.add_(self.anchor_offsets)

        anchor_indices = anchor_indices.view(
            -1,
            2,
        )

        # -----------------------------------------------------
        # Construct anchor weights directly in the final
        # [batch_size, n_features, 2] tensor.
        #
        # Initially:
        #
        #     column 0 = position
        #     column 1 = position
        #
        # Then column 0 is converted to:
        #
        #     1 - position, if a left anchor exists;
        #     0,            for the first bin.
        #
        # expand() itself does not allocate memory.
        # clone() allocates exactly the final output tensor.
        # -----------------------------------------------------

        anchor_weights = (
            position
            .unsqueeze(-1)
            .expand(
                -1,
                -1,
                2,
            )
            .clone()
        )

        left_weight = anchor_weights[..., 0]

        left_weight.neg_()
        left_weight.add_(1.0)

        left_weight.masked_fill_(
            local_bin.eq(0),
            0.0,
        )

        anchor_weights = anchor_weights.view(
            -1,
            2,
        )

        x_ple = F.embedding_bag(
            input=anchor_indices,
            weight=right_anchors,
            offsets=None,
            mode="sum",
            per_sample_weights=anchor_weights,
            sparse=False,
        ).view(
            batch_size,
            self.n_features,
            self.d_embedding,
        )

        if self.linear.bias is not None:
            x_ple = x_ple + self.linear.bias

        if self.activation is not None:
            x_ple = self.activation(x_ple)

        if self.linear0 is None:
            return x_ple

        x_linear = self.linear0(x)

        return x_linear + x_ple