import torch
import torch.nn.functional as F

from .base import BaseOP


class Linear(BaseOP):
    def __init__(self, input_size: int, output_size: int, has_bias: bool = False):
        self.weight = torch.empty(output_size, input_size)
        if has_bias:
            self.bias = torch.empty(output_size)
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)
