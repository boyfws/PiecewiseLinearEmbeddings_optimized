import warnings
from typing import Literal, Optional

import torch
from torch import Tensor, nn
from torch.nn.parameter import Parameter
from rtdl_num_embeddings import _check_bins
import torch.nn.functional as F


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
        # Padded edges are required only by torch.searchsorted.
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

        for feature_idx, edges in enumerate(bins):
            edges_float32 = edges.to(
                device=base_device,
                dtype=torch.float32,
            )

            padded_edges[
                feature_idx,
                :edges_float32.numel(),
            ] = edges_float32

        self.register_buffer(
            "bin_edges",
            padded_edges,
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

        anchor_offsets_list: list[int] = []

        current_offset = 0

        for feature_n_bins in n_bins_list:
            anchor_offsets_list.append(
                current_offset
            )

            current_offset += feature_n_bins

        self.register_buffer(
            "anchor_offsets",
            torch.tensor(
                anchor_offsets_list,
                dtype=torch.int32,
                device=base_device,
            ),
        )

        # -----------------------------------------------------
        # Trainable right anchors.
        #
        # There is no padding and no fixed zero anchor stored here.
        # The left anchor of the first bin is implicitly zero.
        # -----------------------------------------------------

        self.anchors = Parameter(
            torch.empty(
                self.total_n_bins,
                d_embedding,
                device=base_device,
            )
        )

        is_version_b = version == "B"

        self.bias: Optional[Parameter]

        if is_version_b:
            self.register_parameter(
                "bias",
                None,
            )

            self.linear0 = _FeatureLinearEmbedding(
                self.n_features,
                d_embedding,
            )
        else:
            self.bias = Parameter(
                torch.empty(
                    self.n_features,
                    d_embedding,
                    device=base_device,
                )
            )

            self.linear0 = None

        self.activation = (
            nn.ReLU()
            if activation
            else None
        )

        self.reset_parameters()

    def _apply(self, fn):
        """
        Apply device/dtype transformations while keeping bin edges in float32.

        Calling module.half() or module.bfloat16() changes trainable parameters,
        but the bin search remains in float32.
        """
        super()._apply(fn)

        self.bin_edges = self.bin_edges.float()

        return self

    def reset_parameters(self) -> None:
        """
        Initialize anchors through random bin increments.

        This preserves the functional initialization style of the original
        implementation, where every bin stores an independent increment.
        """
        if self.version == "B":
            nn.init.zeros_(self.anchors)

        else:
            bound = self.max_n_bins ** -0.5

            with torch.no_grad():
                current_offset = 0

                for feature_n_bins in self._n_bins_list:
                    increments = torch.empty(
                        feature_n_bins,
                        self.d_embedding,
                        device=self.anchors.device,
                        dtype=self.anchors.dtype,
                    )

                    nn.init.uniform_(
                        increments,
                        -bound,
                        bound,
                    )

                    anchors = torch.cumsum(
                        increments,
                        dim=0,
                    )

                    self.anchors[
                        current_offset:
                        current_offset + feature_n_bins
                    ].copy_(anchors)

                    current_offset += feature_n_bins

            assert self.bias is not None

            nn.init.uniform_(
                self.bias,
                -bound,
                bound,
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
            .to(dtype=torch.float32)
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
        # Select the left and right numerical bin edges.
        #
        # Flattened indexing allows keeping int32 indices.
        # -----------------------------------------------------

        left_edge_indices = (
            local_bin_by_feature
            + self.edge_offsets.unsqueeze(1)
        )

        flat_edges = self.bin_edges.reshape(-1)

        left_edges = torch.index_select(
            flat_edges,
            dim=0,
            index=left_edge_indices.reshape(-1),
        ).view(
            self.n_features,
            batch_size,
        )

        right_edges = torch.index_select(
            flat_edges,
            dim=0,
            index=(
                left_edge_indices.reshape(-1) + 1
            ),
        ).view(
            self.n_features,
            batch_size,
        )

        # t may be below zero or above one for extrapolation.
        position_by_feature = (
            values_by_feature - left_edges
        ) / (
            right_edges - left_edges
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
            .to(dtype=self.anchors.dtype)
        )

        # [batch_size, n_features]
        global_right_indices = local_bin + self.anchor_offsets

        has_left_anchor = local_bin.ne(0)

        global_left_indices = (
            global_right_indices
            - has_left_anchor.to(global_right_indices.dtype)
        )
        anchor_indices = torch.stack(
            (
                global_left_indices,
                global_right_indices,
            ),
            dim=-1,
        ).reshape(-1, 2)

        # Вес левого anchor:
        #   (1 - position), если bin > 0
        #   0,              если bin == 0
        #
        # Вес правого anchor:
        #   position
        left_weight = (
            (1.0 - position)
            * has_left_anchor.to(position.dtype)
        )

        anchor_weights = torch.stack(
            (
                left_weight,
                position,
            ),
            dim=-1,
        ).reshape(-1, 2)

        anchor_weights = anchor_weights.to(self.anchors.dtype)

        x_ple = F.embedding_bag(
            input=anchor_indices,
            weight=self.anchors,
            offsets=None,
            mode="sum",
            per_sample_weights=anchor_weights,
            sparse=False,
        ).view(
            batch_size,
            self.n_features,
            self.d_embedding,
        )

        if self.bias is not None:
            x_ple = x_ple + self.bias

        if self.activation is not None:
            x_ple = self.activation(x_ple)

        if self.linear0 is None:
            return x_ple

        x_linear = self.linear0(x)

        return x_linear + x_ple