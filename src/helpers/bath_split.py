from typing import Any

import torch
from torch import Tensor

from src.model.hyper_paramters import BATCH_SIZE, BLOCK_SIZE


def get_batches(split: dict[str, Any], spit_type: str) -> tuple[Tensor, Tensor]:
    data = split["train"] if spit_type == "train" else split["validation"]
    radom_index_input = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))

    return torch.stack(
        [data[i : i + BLOCK_SIZE] for i in radom_index_input]
    ), torch.stack([data[i + 1 : i + BLOCK_SIZE + 1] for i in radom_index_input])
