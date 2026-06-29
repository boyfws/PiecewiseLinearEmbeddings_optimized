from torch import Tensor
import torch
from typing import Sequence

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
