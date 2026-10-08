set -e

uv run python benchmark.py \
  --model /mnt/afs/models/Qwen/Qwen3-0.6B \
  --device cuda:0 \
  --dtype bfloat16 \
  --max-new-tokens 128 \
  --warmup-runs 1 \
  --runs 3 \
  --ignore-eos \
  --use-kv-cache \
  --json-out docs/benchmark-cuda.json
