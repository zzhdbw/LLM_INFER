set -e

uv run python main.py \
  --model /mnt/afs/models/Qwen/Qwen3-0.6B \
  --prompt "你好" \
  --device cuda:0 \
  --dtype auto
