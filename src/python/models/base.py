from abc import ABC, abstractmethod

import torch

from ..layers.base import BaseOP


class BaseLLMModel(ABC, BaseOP):
    @abstractmethod
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor: ...
