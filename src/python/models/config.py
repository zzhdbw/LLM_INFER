from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any


def _parse_rope_parameters(
    rope_parameters: dict[str, Any] | None,
    rope_theta: float | None = None,
) -> tuple[float, str, dict[str, Any] | None]:
    if not isinstance(rope_parameters, dict):
        rope_parameters = {}

    if rope_theta is None:
        rope_theta = rope_parameters.get("rope_theta", 1000000.0)

    rope_type = rope_parameters.get(
        "rope_type",
        rope_parameters.get("type", "default"),
    )

    rope_scaling = {
        key: value
        for key, value in rope_parameters.items()
        if key not in {"rope_theta", "rope_type", "type"}
    } or None

    return float(rope_theta), str(rope_type), rope_scaling


@dataclass(frozen=True)
class ModelConfig:
    num_layers: int
    num_qo_heads: int
    num_kv_heads: int
    head_dim: int
    hidden_size: int
    vocab_size: int
    intermediate_size: int
    hidden_act: str
    rms_norm_eps: float
    rope_theta: float
    max_position_embeddings: int
    tie_word_embeddings: bool
    rope_type: str = "default"  # TODO
    rope_scaling: dict[str, Any] | None = None  # TODO

    @classmethod
    def from_hf(cls, config) -> ModelConfig:
        """Load from a transformers config object (aligned with mini-sglang)."""
        num_kv_heads = getattr(
            config, "num_key_value_heads", config.num_attention_heads
        )
        head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )

        rope_parameters = (
            getattr(config, "rope_parameters", None)
            or getattr(config, "rope_scaling", None)
            or {}
        )
        rope_theta, rope_type, rope_scaling = _parse_rope_parameters(
            rope_parameters,
            getattr(config, "rope_theta", None),
        )

        return cls(
            num_layers=config.num_hidden_layers,
            num_qo_heads=config.num_attention_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            hidden_size=config.hidden_size,
            vocab_size=config.vocab_size,
            intermediate_size=config.intermediate_size,
            hidden_act=getattr(config, "hidden_act", "silu"),
            rms_norm_eps=config.rms_norm_eps,
            rope_theta=rope_theta,
            max_position_embeddings=config.max_position_embeddings,
            tie_word_embeddings=getattr(config, "tie_word_embeddings", False),
            rope_type=rope_type,
            rope_scaling=rope_scaling,
        )

    @classmethod
    def from_json(cls, model_path: str) -> ModelConfig:
        """Load from config.json file (fallback)."""
        config_path = os.path.join(model_path, "config.json")
        with open(config_path) as f:
            data = json.load(f)

        rope_parameters = data.get("rope_parameters") or data.get("rope_scaling") or {}
        rope_theta, rope_type, rope_scaling = _parse_rope_parameters(
            rope_parameters,
            data.get("rope_theta"),
        )

        return cls(
            num_layers=data["num_hidden_layers"],
            num_qo_heads=data["num_attention_heads"],
            num_kv_heads=data["num_key_value_heads"],
            head_dim=data.get(
                "head_dim", data["hidden_size"] // data["num_attention_heads"]
            ),
            hidden_size=data["hidden_size"],
            vocab_size=data["vocab_size"],
            intermediate_size=data["intermediate_size"],
            hidden_act=data.get("hidden_act", "silu"),
            rms_norm_eps=data.get("rms_norm_eps", 1e-6),
            rope_theta=rope_theta,
            max_position_embeddings=data.get("max_position_embeddings", 32768),
            tie_word_embeddings=data.get("tie_word_embeddings", False),
            rope_type=rope_type,
            rope_scaling=rope_scaling,
        )
