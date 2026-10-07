from .base import BaseLLMModel
from .config import ModelConfig
from .qwen3 import (
    Qwen3Attention,
    Qwen3DecoderLayer,
    Qwen3ForCausalLM,
    Qwen3MLP,
    Qwen3Model,
)
from .weight import load_weights


def create_model(model_path: str, config: ModelConfig) -> BaseLLMModel:
    model_name = model_path.lower()
    if "qwen3" in model_name:
        from .qwen3 import Qwen3ForCausalLM

        return Qwen3ForCausalLM(config)

    raise ValueError(f"Unsupported model: {model_path}")


__all__ = [
    "BaseLLMModel",
    "ModelConfig",
    "Qwen3Attention",
    "Qwen3DecoderLayer",
    "Qwen3ForCausalLM",
    "Qwen3MLP",
    "Qwen3Model",
    "load_weights",
]
