set -e

uv run python main.py \
  --model /mnt/afs/models/Qwen/Qwen3-0.6B \
  --prompt "写一个长篇文章论证亚洲和欧洲的气候不同的原因" \
  --device cuda:0 \
  --dtype auto \
  --max-tokens 3000 \
  --no-use-kv-cache
