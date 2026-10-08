import argparse
import time

import torch

from src.python import LLM, SamplingParams

_DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def resolve_device(device: str | None) -> torch.device:
    if device:
        return torch.device(device)
    if torch.cuda.is_available():
        free = [torch.cuda.mem_get_info(i)[0] for i in range(torch.cuda.device_count())]
        best = free.index(max(free))
        return torch.device(f"cuda:{best}")
    return torch.device("cpu")


def resolve_dtype(dtype: str, device: torch.device) -> torch.dtype:
    if dtype == "auto":
        return torch.bfloat16 if device.type == "cuda" else torch.float32
    return _DTYPE_MAP[dtype]


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen3 inference")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="HuggingFace model name or local model directory",
    )
    parser.add_argument("--prompt", type=str, default="你好")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "bfloat16", "float16", "float32"],
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device, e.g. cuda:0, cuda:1, cpu. Default: auto.",
    )
    parser.add_argument(
        "--use-kv-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="启用 KV cache。使用 --no-use-kv-cache 关闭。",
    )
    args = parser.parse_args()

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)

    print("=" * 60)
    print("Qwen3 inference")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Dtype : {dtype}")
    print(f"Use KV cache: {args.use_kv_cache}")

    llm = LLM(args.model, dtype=dtype, device=str(device))
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )

    print(f"\nPrompt: {args.prompt}")
    print(f"Max tokens: {args.max_tokens}")

    t0 = time.perf_counter()
    results = llm.generate(
        [args.prompt], sampling_params, use_kv_cache=args.use_kv_cache
    )
    elapsed = time.perf_counter() - t0

    output = results[0]
    print("\n" + "=" * 60)
    print(output["text"])
    print("=" * 60)
    print(f"Generated tokens: {len(output['token_ids'])}")
    print(f"Total time: {elapsed:.2f}s")
    if output["token_ids"]:
        print(f"Throughput: {len(output['token_ids']) / elapsed:.2f} tok/s")


if __name__ == "__main__":
    main()
