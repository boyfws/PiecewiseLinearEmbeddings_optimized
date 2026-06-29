import torch
from torch import Tensor


BIN_CASE_NAMES = (
    "equal",
    "ragged",
    "single_bin",
    "all_single_bin",
)


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
