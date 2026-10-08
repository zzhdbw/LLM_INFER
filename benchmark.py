import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.python import LLM, SamplingParams
from src.python.engine import Sampler

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


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def build_inputs(llm: LLM, prompt: str, device: torch.device) -> torch.Tensor:
    messages = [{"role": "user", "content": prompt}]
    text = llm.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return llm.tokenizer.encode(text, return_tensors="pt").to(device)


def run_generate(
    llm: LLM,
    input_ids: torch.Tensor,
    sampling_params: SamplingParams,
    device: torch.device,
    use_kv_cache: bool,
    max_seq_len: int,
) -> dict[str, Any]:
    if input_ids.shape[1] + sampling_params.max_tokens > max_seq_len:
        raise ValueError(
            f"prompt length ({input_ids.shape[1]}) + max_tokens "
            f"({sampling_params.max_tokens}) exceeds max_seq_len ({max_seq_len})"
        )

    generated = input_ids.clone()
    model_input = input_ids
    kv_cache = (
        llm.create_kv_cache(max_seq_len=max_seq_len, batch_size=input_ids.shape[0])
        if use_kv_cache
        else None
    )
    sampler = Sampler(sampling_params)
    step_times: list[float] = []

    synchronize(device)
    t0 = time.perf_counter()

    for _ in range(sampling_params.max_tokens):
        synchronize(device)
        step_start = time.perf_counter()
        logits = llm.model.forward(model_input, kv_cache)
        synchronize(device)
        step_times.append(time.perf_counter() - step_start)

        next_logits = logits[:, -1, :]
        next_token = sampler.sample(next_logits)

        generated = torch.cat([generated, next_token], dim=-1)
        if (
            not sampling_params.ignore_eos
            and next_token.item() == llm.tokenizer.eos_token_id
        ):
            break

        if use_kv_cache:
            model_input = next_token
        else:
            model_input = generated

    synchronize(device)
    total_s = time.perf_counter() - t0

    new_ids = generated[0][input_ids.shape[1] :]
    generated_tokens = int(new_ids.shape[0])
    output_text = llm.tokenizer.decode(new_ids, skip_special_tokens=True)

    prefill_ms = step_times[0] * 1000 if step_times else 0.0
    decode_times = step_times[1:]
    decode_avg_ms = statistics.mean(decode_times) * 1000 if decode_times else 0.0
    decode_tokens_per_s = 1000.0 / decode_avg_ms if decode_avg_ms > 0 else 0.0

    return {
        "input_tokens": int(input_ids.shape[1]),
        "generated_tokens": generated_tokens,
        "total_s": total_s,
        "prefill_ms": prefill_ms,
        "decode_avg_ms": decode_avg_ms,
        "decode_tokens_per_s": decode_tokens_per_s,
        "overall_tokens_per_s": generated_tokens / total_s if total_s > 0 else 0.0,
        "use_kv_cache": use_kv_cache,
        "max_seq_len": max_seq_len,
        "output": output_text,
    }


def mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Qwen3 inference")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "bfloat16", "float16", "float32"],
    )
    parser.add_argument("--prompt", default="你好")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=4096,
        help="预分配 KV cache 的最大序列长度",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument(
        "--ignore-eos",
        action="store_true",
        help="继续生成直到 max_new_tokens，即使遇到 EOS。",
    )
    parser.add_argument(
        "--use-kv-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="启用 KV cache。使用 --no-use-kv-cache 关闭。",
    )
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        ignore_eos=args.ignore_eos,
        max_tokens=args.max_new_tokens,
    )

    print("=" * 72)
    print("Qwen3 inference benchmark")
    print("=" * 72)
    print(f"model        : {args.model}")
    print(f"device       : {device}")
    print(f"dtype        : {dtype}")
    print(f"torch        : {torch.__version__}")
    print(f"python       : {platform.python_version()}")
    print(f"platform     : {platform.platform()}")
    print(f"processor    : {platform.processor()}")
    print(f"max_new_tokens: {args.max_new_tokens}")
    print(f"max_seq_len   : {args.max_seq_len}")
    print(f"temperature  : {args.temperature}")
    print(f"top_k        : {args.top_k}")
    print(f"top_p        : {args.top_p}")
    print(f"warmup_runs  : {args.warmup_runs}")
    print(f"runs         : {args.runs}")
    print(f"ignore_eos   : {args.ignore_eos}")
    print(f"use_kv_cache : {args.use_kv_cache}")

    llm = LLM(args.model, dtype=dtype, device=str(device))
    num_params = sum(p.numel() for p in llm.model.state_dict().values())

    input_ids = build_inputs(llm, args.prompt, device)
    print(f"parameters   : {num_params:,}")
    print(f"input_tokens : {input_ids.shape[1]}")

    for _ in range(args.warmup_runs):
        run_generate(
            llm,
            input_ids,
            sampling_params,
            device,
            args.use_kv_cache,
            args.max_seq_len,
        )

    runs = [
        run_generate(
            llm,
            input_ids,
            sampling_params,
            device,
            args.use_kv_cache,
            args.max_seq_len,
        )
        for _ in range(args.runs)
    ]

    print()
    print(
        "| Run | Total (s) | Generated | Prefill (ms) | Decode avg (ms/tok) | Decode tok/s | Overall tok/s |"
    )
    print("|---:|---:|---:|---:|---:|---:|---:|")
    for index, run in enumerate(runs, start=1):
        print(
            f"| {index} "
            f"| {run['total_s']:.3f} "
            f"| {run['generated_tokens']} "
            f"| {run['prefill_ms']:.1f} "
            f"| {run['decode_avg_ms']:.1f} "
            f"| {run['decode_tokens_per_s']:.2f} "
            f"| {run['overall_tokens_per_s']:.2f} |"
        )

    summary = {
        "total_s": mean([run["total_s"] for run in runs]),
        "prefill_ms": mean([run["prefill_ms"] for run in runs]),
        "decode_avg_ms": mean([run["decode_avg_ms"] for run in runs]),
        "decode_tokens_per_s": mean([run["decode_tokens_per_s"] for run in runs]),
        "overall_tokens_per_s": mean([run["overall_tokens_per_s"] for run in runs]),
        "use_kv_cache": args.use_kv_cache,
        "max_seq_len": args.max_seq_len,
        "generated_tokens": mean([run["generated_tokens"] for run in runs]),
    }

    print()
    print("Summary")
    print(f"  total             : {summary['total_s']:.3f} s")
    print(f"  generated tokens  : {summary['generated_tokens']:.1f}")
    print(f"  prefill           : {summary['prefill_ms']:.1f} ms")
    print(f"  decode avg        : {summary['decode_avg_ms']:.1f} ms/token")
    print(f"  decode throughput : {summary['decode_tokens_per_s']:.2f} tok/s")
    print(f"  overall throughput: {summary['overall_tokens_per_s']:.2f} tok/s")
    print()
    print("Last output:")
    print(runs[-1]["output"])

    if args.json_out:
        payload = {
            "model": args.model,
            "device": str(device),
            "dtype": str(dtype),
            "torch": torch.__version__,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "parameters": num_params,
            "input_tokens": int(input_ids.shape[1]),
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "ignore_eos": args.ignore_eos,
            "use_kv_cache": args.use_kv_cache,
            "max_seq_len": args.max_seq_len,
            "warmup_runs": args.warmup_runs,
            "runs": runs,
            "summary": summary,
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\nJSON written: {args.json_out}")


if __name__ == "__main__":
    main()
