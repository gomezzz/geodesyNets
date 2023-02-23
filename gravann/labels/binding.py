from typing import Callable

import torch


def bind_label(method: str) -> Callable[[torch.Tensor], torch.Tensor]:
    pass
