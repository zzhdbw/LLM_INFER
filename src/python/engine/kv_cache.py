from __future__ import annotations

from typing import Protocol

import torch


class KVCache(Protocol):
    """Protocol implemented by per-layer K/V caches."""

    def get_seq_len(self) -> int: ...

    def update(
        self,
        layer_idx: int,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

    def clear(self) -> None: ...


class DynamicKVCache:
    """A simple per-layer KV cache that grows with ``torch.cat``.

    Each layer stores:
    - key cache:   (batch, num_kv_heads, seq_len_so_far, head_dim)
    - value cache: (batch, num_kv_heads, seq_len_so_far, head_dim)

    This matches the core idea behind Hugging Face's dynamic cache, but keeps
    the implementation intentionally small and easy to read.
    """

    def __init__(self, num_layers: int):

        self.key_cache: list[torch.Tensor | None] = [None] * num_layers
        self.value_cache: list[torch.Tensor | None] = [None] * num_layers
        self._seq_len = 0

    def get_seq_len(self) -> int:
        return self._seq_len

    def update(
        self,
        layer_idx: int,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cached_keys = self.key_cache[layer_idx]
        cached_values = self.value_cache[layer_idx]

        if cached_keys is None:
            full_keys = key_states
            full_values = value_states
        else:
            full_keys = torch.cat([cached_keys, key_states], dim=2)
            full_values = torch.cat([cached_values, value_states], dim=2)

        self.key_cache[layer_idx] = full_keys
        self.value_cache[layer_idx] = full_values

        # All layers grow by the same amount every decoding step, so tracking
        # the first layer is enough to know the current cached sequence length.
        if layer_idx == 0:
            self._seq_len = full_keys.shape[2]

        return full_keys, full_values

    def clear(self) -> None:
        for layer_idx in range(len(self.key_cache)):
            self.key_cache[layer_idx] = None
            self.value_cache[layer_idx] = None
        self._seq_len = 0


class PreallocatedKVCache:
    """Per-layer KV cache with preallocated K/V tensors.

    The cache is allocated once as:

    - key cache:   (batch, num_kv_heads, max_seq_len, head_dim)
    - value cache: (batch, num_kv_heads, max_seq_len, head_dim)

    During inference, newly computed K/V states are written in place.  This
    avoids the per-step ``torch.cat`` and allocation overhead of
    :class:`DynamicKVCache`.
    """

    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        max_seq_len: int,
        batch_size: int = 1,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ):
        if max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be positive, got {max_seq_len}")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size

        cache_shape = (batch_size, num_kv_heads, max_seq_len, head_dim)
        self.key_cache = [
            torch.empty(cache_shape, device=device, dtype=dtype)
            for _ in range(num_layers)
        ]
        self.value_cache = [
            torch.empty(cache_shape, device=device, dtype=dtype)
            for _ in range(num_layers)
        ]

        self._seq_len = 0

    @property
    def seq_len(self) -> int:
        return self._seq_len

    def get_seq_len(self) -> int:
        return self._seq_len

    def update(
        self,
        layer_idx: int,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        start = self._seq_len
        end = start + key_states.shape[2]
        if end > self.max_seq_len:
            raise RuntimeError(
                f"KV cache overflow: need sequence length {end}, "
                f"but max_seq_len is {self.max_seq_len}. "
                "Increase max_seq_len or reduce --max-tokens."
            )

        self.key_cache[layer_idx][:, :, start:end, :] = key_states
        self.value_cache[layer_idx][:, :, start:end, :] = value_states

        # All layers write the same token range in one forward, so only the
        # last layer advances the global sequence length.
        if layer_idx == self.num_layers - 1:
            self._seq_len = end

        return (
            self.key_cache[layer_idx][:, :, :end, :],
            self.value_cache[layer_idx][:, :, :end, :],
        )

    def clear(self) -> None:
        self._seq_len = 0
