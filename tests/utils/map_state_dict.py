from collections import OrderedDict
from collections.abc import Mapping

import torch
from torch import Tensor


def convert_old_ple_state_dict(
    old_state_dict: Mapping[str, Tensor],
    *,
    old_prefix: str = "",
    new_prefix: str = "",
) -> OrderedDict[str, Tensor]:
    """
    Convert the original PiecewiseLinearEmbeddings state dict to the
    anchor-based SearchsortedPiecewiseLinearEmbeddings format.

    The original implementation stores trainable bin increments in:

        linear.weight: [n_features, max_n_bins, d_embedding]

    The new implementation stores cumulative right anchors in a compact
    ragged tensor:

        anchors: [sum(n_bins_per_feature), d_embedding]

    For each feature:

        anchor[j] = sum(increment[k] for k <= j)

    The number of real bins is inferred automatically:

        - if ``impl.mask`` exists, the number of True values in each row
          equals the number of real bins for that feature;
        - if ``impl.mask`` is absent, all features are assumed to have
          ``max_n_bins`` bins.

    Only parameters required by the new implementation are returned.
    Old PLE encoding buffers, masks, padding parameters, and unrelated
    state-dict entries are intentionally discarded.

    Args:
        old_state_dict:
            The state dict of the original PLE module, or a larger model
            state dict when ``old_prefix`` is provided.

        old_prefix:
            Prefix of the original PLE module in ``old_state_dict``.

        new_prefix:
            Prefix to prepend to keys in the returned state dict.

    Returns:
        An OrderedDict containing only the state required by the new module.

        Version A:
            anchors
            bias

        Version B:
            anchors
            linear0.weight
            linear0.bias

    Notes:
        Optimizer state cannot be converted directly because the old and new
        implementations use different parameterizations.
    """

    def old_key(name: str) -> str:
        return f"{old_prefix}{name}"

    def new_key(name: str) -> str:
        return f"{new_prefix}{name}"

    weight_key = old_key("linear.weight")

    if weight_key not in old_state_dict:
        available_keys = list(old_state_dict.keys())

        raise KeyError(
            f"Missing required key {weight_key!r}. "
            f"Available keys include: {available_keys[:20]}"
        )

    old_weight = old_state_dict[weight_key]

    if not isinstance(old_weight, Tensor):
        raise TypeError(
            f"{weight_key!r} must contain a Tensor, "
            f"got {type(old_weight)}"
        )

    if old_weight.ndim != 3:
        raise ValueError(
            f"{weight_key!r} must have shape "
            "[n_features, max_n_bins, d_embedding], "
            f"got {tuple(old_weight.shape)}"
        )

    n_features, max_n_bins, d_embedding = old_weight.shape

    if n_features <= 0:
        raise ValueError("The number of features must be positive")

    if max_n_bins <= 0:
        raise ValueError("The maximum number of bins must be positive")

    if d_embedding <= 0:
        raise ValueError("The embedding dimension must be positive")

    # ---------------------------------------------------------
    # Detect the original PLE version.
    # ---------------------------------------------------------

    linear_bias_key = old_key("linear.bias")
    linear0_weight_key = old_key("linear0.weight")
    linear0_bias_key = old_key("linear0.bias")

    has_linear_bias = linear_bias_key in old_state_dict
    has_linear0_weight = linear0_weight_key in old_state_dict
    has_linear0_bias = linear0_bias_key in old_state_dict

    if has_linear0_weight != has_linear0_bias:
        raise ValueError(
            "The original state dict contains only one of "
            "'linear0.weight' and 'linear0.bias'"
        )

    if has_linear0_weight:
        version = "B"

        if has_linear_bias:
            raise ValueError(
                "The state dict is inconsistent: version B contains "
                "linear0 parameters, but linear.bias is also present"
            )

    else:
        version = "A"

        if not has_linear_bias:
            raise ValueError(
                "Could not determine the original PLE version: neither "
                "linear.bias nor linear0 parameters are present"
            )

    # ---------------------------------------------------------
    # Infer the number of real bins for every feature.
    #
    # impl.mask is absent when all features have the same number
    # of bins.
    # ---------------------------------------------------------

    mask_key = old_key("impl.mask")
    old_mask = old_state_dict.get(mask_key)

    if old_mask is None:
        n_bins_per_feature = (
            max_n_bins,
        ) * n_features

    else:
        if not isinstance(old_mask, Tensor):
            raise TypeError(
                f"{mask_key!r} must contain a Tensor, "
                f"got {type(old_mask)}"
            )

        expected_mask_shape = (
            n_features,
            max_n_bins,
        )

        if tuple(old_mask.shape) != expected_mask_shape:
            raise ValueError(
                f"{mask_key!r} must have shape "
                f"{expected_mask_shape}, "
                f"got {tuple(old_mask.shape)}"
            )

        n_bins_per_feature = tuple(
            int(value)
            for value in (
                old_mask
                .to(
                    device="cpu",
                    dtype=torch.int64,
                )
                .sum(dim=1)
                .tolist()
            )
        )

    for feature_idx, feature_n_bins in enumerate(
        n_bins_per_feature
    ):
        if not 1 <= feature_n_bins <= max_n_bins:
            raise ValueError(
                f"Invalid number of bins for feature {feature_idx}: "
                f"{feature_n_bins}. Expected a value in "
                f"[1, {max_n_bins}]"
            )

    # ---------------------------------------------------------
    # Convert old physical increments to compact logical anchors.
    #
    # For a feature with fewer than max_n_bins bins, the original
    # layout stores the last real bin at the final physical index:
    #
    # logical:
    #     W0, W1, W2
    #
    # physical:
    #     W0, W1, padding, padding, W2
    # ---------------------------------------------------------

    anchor_chunks: list[Tensor] = []

    for feature_idx, feature_n_bins in enumerate(
        n_bins_per_feature
    ):
        feature_weight = old_weight[feature_idx]

        if feature_n_bins == 1:
            logical_increments = feature_weight[-1:]

        elif feature_n_bins == max_n_bins:
            logical_increments = feature_weight

        else:
            logical_increments = torch.cat(
                [
                    feature_weight[
                        :feature_n_bins - 1
                    ],
                    feature_weight[-1:],
                ],
                dim=0,
            )

        feature_anchors = torch.cumsum(
            logical_increments,
            dim=0,
        )

        anchor_chunks.append(
            feature_anchors
        )

    anchors = torch.cat(
        anchor_chunks,
        dim=0,
    )

    expected_anchor_shape = (
        sum(n_bins_per_feature),
        d_embedding,
    )

    if tuple(anchors.shape) != expected_anchor_shape:
        raise RuntimeError(
            "Internal conversion error: "
            f"expected anchors with shape {expected_anchor_shape}, "
            f"got {tuple(anchors.shape)}"
        )

    converted_state_dict: OrderedDict[str, Tensor] = (
        OrderedDict()
    )

    converted_state_dict[
        new_key("anchors")
    ] = anchors.detach().clone()

    # ---------------------------------------------------------
    # Copy only parameters required by the detected version.
    # ---------------------------------------------------------

    if version == "A":
        old_bias = old_state_dict[
            linear_bias_key
        ]

        expected_bias_shape = (
            n_features,
            d_embedding,
        )

        if tuple(old_bias.shape) != expected_bias_shape:
            raise ValueError(
                f"{linear_bias_key!r} must have shape "
                f"{expected_bias_shape}, "
                f"got {tuple(old_bias.shape)}"
            )

        converted_state_dict[
            new_key("bias")
        ] = old_bias.detach().clone()

    else:
        old_linear0_weight = old_state_dict[
            linear0_weight_key
        ]

        old_linear0_bias = old_state_dict[
            linear0_bias_key
        ]

        expected_linear0_shape = (
            n_features,
            d_embedding,
        )

        if (
            tuple(old_linear0_weight.shape)
            != expected_linear0_shape
        ):
            raise ValueError(
                f"{linear0_weight_key!r} must have shape "
                f"{expected_linear0_shape}, "
                f"got {tuple(old_linear0_weight.shape)}"
            )

        if (
            tuple(old_linear0_bias.shape)
            != expected_linear0_shape
        ):
            raise ValueError(
                f"{linear0_bias_key!r} must have shape "
                f"{expected_linear0_shape}, "
                f"got {tuple(old_linear0_bias.shape)}"
            )

        converted_state_dict[
            new_key("linear0.weight")
        ] = old_linear0_weight.detach().clone()

        converted_state_dict[
            new_key("linear0.bias")
        ] = old_linear0_bias.detach().clone()

    return converted_state_dict