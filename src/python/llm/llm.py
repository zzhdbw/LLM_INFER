import os

import torch
from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoTokenizer

from ..core import SamplingParams
from ..engine import PreallocatedKVCache, Sampler
from ..models import ModelConfig, create_model, load_weights


def _resolve_model_path(model_path: str) -> str:
    """Resolve a HuggingFace hub ID to a local path if needed."""
    if os.path.isdir(model_path):
        return model_path
    return snapshot_download(model_path)


class LLM:
    def __init__(
        self,
        model_path: str,
        dtype: torch.dtype = torch.bfloat16,
        max_seq_len: int = 4096,
        **kwargs,
    ):
        self.device = torch.device(kwargs.get("device", "cuda"))
        self.dtype = dtype

        model_path = _resolve_model_path(model_path)
        hf_config = AutoConfig.from_pretrained(model_path)
        config = ModelConfig.from_hf(hf_config)
        if max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be positive, got {max_seq_len}")
        if max_seq_len > config.max_position_embeddings:
            raise ValueError(
                f"max_seq_len ({max_seq_len}) exceeds model maximum "
                f"position embeddings ({config.max_position_embeddings})"
            )

        self.config = config
        self.max_seq_len = max_seq_len
        self.num_layers = config.num_layers
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        previous_dtype = torch.get_default_dtype()
        torch.set_default_dtype(self.dtype)
        try:
            with torch.device("meta"):
                self.model = create_model(model_path, config)
        finally:
            torch.set_default_dtype(previous_dtype)

        load_weights(self.model, model_path, self.device, self.dtype)

        # Move rotary embedding cache to device
        self.model.model._rotary_emb.set_device(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

    def create_kv_cache(
        self,
        max_seq_len: int | None = None,
        batch_size: int = 1,
    ) -> PreallocatedKVCache:
        if max_seq_len is None:
            max_seq_len = self.max_seq_len
        if max_seq_len > self.config.max_position_embeddings:
            raise ValueError(
                f"max_seq_len ({max_seq_len}) exceeds model maximum "
                f"position embeddings ({self.config.max_position_embeddings})"
            )
        return PreallocatedKVCache(
            num_layers=self.num_layers,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            max_seq_len=max_seq_len,
            batch_size=batch_size,
            device=self.device,
            dtype=self.dtype,
        )

    @torch.no_grad()
    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams] | None = None,
        use_kv_cache: bool = True,
        max_seq_len: int | None = None,
    ) -> list[dict]:
        if sampling_params is None:
            sampling_params = SamplingParams()

        if max_seq_len is None:
            max_seq_len = self.max_seq_len

        # Normalize to per-request sampling params list
        if isinstance(sampling_params, SamplingParams):
            params_list = [sampling_params] * len(prompts)
        else:
            params_list = sampling_params

        results = []
        for prompt, sp in zip(prompts, params_list):
            sampler = Sampler(sp)

            if isinstance(prompt, str):
                messages = [{"role": "user", "content": prompt}]
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                input_ids = self.tokenizer.encode(text, return_tensors="pt").to(
                    self.device
                )
            else:
                input_ids = torch.tensor([prompt], device=self.device)

            if input_ids.shape[1] + sp.max_tokens > max_seq_len:
                raise ValueError(
                    f"prompt length ({input_ids.shape[1]}) + max_tokens "
                    f"({sp.max_tokens}) exceeds max_seq_len ({max_seq_len}). "
                    "Increase --max-seq-len or reduce --max-tokens."
                )

            generated = input_ids.clone()
            model_input = input_ids

            if use_kv_cache:
                kv_cache = self.create_kv_cache(
                    max_seq_len=max_seq_len,
                    batch_size=input_ids.shape[0],
                )
            else:
                kv_cache = None

            for _ in range(sp.max_tokens):
                logits = self.model.forward(model_input, kv_cache)
                next_logits = logits[:, -1, :]
                next_token = sampler.sample(next_logits)

                generated = torch.cat([generated, next_token], dim=-1)
                if (
                    not sp.ignore_eos
                    and next_token.item() == self.tokenizer.eos_token_id
                ):
                    break

                if use_kv_cache:
                    model_input = next_token
                else:
                    model_input = generated

            new_token_ids = generated[0][input_ids.shape[1] :].tolist()
            text = self.tokenizer.decode(new_token_ids, skip_special_tokens=True)
            results.append({"text": text, "token_ids": new_token_ids})

        return results
