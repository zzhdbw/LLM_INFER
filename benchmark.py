import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent
for search_path in (ROOT, ROOT / "src"):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from main import generate, resolve_device, resolve_model_path
from python.models import ModelConfig, Qwen3ForCausalLM, load_weights


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def build_inputs(tokenizer: Any, prompt: str, device: torch.device) -> torch.Tensor:
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return tokenizer.encode(text, return_tensors="pt").to(device)


def run_generate(
    model: Qwen3ForCausalLM,
    tokenizer: Any,
    input_ids: torch.Tensor,
    device: torch.device,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    eos_token_id: int,
) -> dict[str, Any]:
    synchronize(device)
    t0 = time.perf_counter()
    output_ids, stats = generate(
        model,
        input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        eos_token_id=eos_token_id,
    )
    synchronize(device)
    total_s = time.perf_counter() - t0

    new_ids = output_ids[0][input_ids.shape[1] :]
    generated_tokens = int(new_ids.shape[0])
    output_text = tokenizer.decode(new_ids, skip_special_tokens=True)

    step_times = stats["step_times"]
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
        "output": output_text,
    }


def mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Qwen3 inference")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--device", default=None)
    parser.add_argument("--prompt", default="你好")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument(
        "--ignore-eos",
        action="store_true",
        help="继续生成直到 max_new_tokens，即使遇到 EOS。",
    )
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    device = resolve_device(args.device)
    model_path = resolve_model_path(args.model)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    print("=" * 72)
    print("Qwen3 inference benchmark")
    print("=" * 72)
    print(f"model        : {model_path}")
    print(f"device       : {device}")
    print(f"dtype        : {dtype}")
    print(f"torch        : {torch.__version__}")
    print(f"python       : {platform.python_version()}")
    print(f"platform     : {platform.platform()}")
    print(f"processor    : {platform.processor()}")
    print(f"max_new_tokens: {args.max_new_tokens}")
    print(f"warmup_runs  : {args.warmup_runs}")
    print(f"runs         : {args.runs}")
    print(f"ignore_eos   : {args.ignore_eos}")

    torch.set_default_dtype(dtype)
    config = ModelConfig.from_json(str(model_path))
    model = Qwen3ForCausalLM(config)
    load_weights(model, str(model_path), device, dtype)
    model.model._rotary_emb.set_device(device)
    num_params = sum(p.numel() for p in model.state_dict().values())

    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    input_ids = build_inputs(tokenizer, args.prompt, device)
    eos_token_id = -1 if args.ignore_eos else tokenizer.eos_token_id
    print(f"parameters   : {num_params:,}")
    print(f"input_tokens : {input_ids.shape[1]}")

    for _ in range(args.warmup_runs):
        run_generate(
            model,
            tokenizer,
            input_ids,
            device,
            args.max_new_tokens,
            args.temperature,
            args.top_k,
            args.top_p,
            eos_token_id,
        )

    runs = [
        run_generate(
            model,
            tokenizer,
            input_ids,
            device,
            args.max_new_tokens,
            args.temperature,
            args.top_k,
            args.top_p,
            eos_token_id,
        )
        for _ in range(args.runs)
    ]

    print()
    print("| Run | Total (s) | Generated | Prefill (ms) | Decode avg (ms/tok) | Decode tok/s | Overall tok/s |")
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
            "model": str(model_path),
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
            "warmup_runs": args.warmup_runs,
            "ignore_eos": args.ignore_eos,
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
