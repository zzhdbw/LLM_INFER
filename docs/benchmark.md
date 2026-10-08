# Benchmark

本文记录 Qwen3 手写推理实现 v0.1、v0.2 和当前版本在 **NVIDIA A800** 上的 benchmark 结果。

> v0.1 和 v0.2 使用历史 tag 运行
>
> 当前版本包含 KV cache 实现
>
> 所有 GPU 结果均使用项目 `.venv` 中的 `torch 2.8.0+cu128` 和 `transformers 5.19.0`

## 测试环境

### GPU

| 项目 | 值 |
|---|---|
| GPU | NVIDIA A800-SXM4-80GB × 8 |
| 显存 | 81920 MiB / 80 GB |
| 使用设备 | `cuda:0` |
| NVIDIA Driver | 570.148.08 |
| 操作系统 | Linux 6.8.0-60-generic x86_64 |
| Python | 3.11.11 |
| PyTorch | 2.8.0+cu128 |
| transformers | 5.19.0 |
| dtype | `torch.bfloat16` |

### CPU（历史数据）

| 项目 | 值 |
|---|---|
| CPU | AMD EPYC 7742 64-Core Processor × 2 sockets，共 256 逻辑 CPU |
| 内存 | 1.0 TiB |
| 操作系统 | Linux 5.15.0-140-generic x86_64 |
| Python | 3.11.11 |
| PyTorch | 2.8.0+cu128 |
| CUDA | 未启用，使用 CPU |
| transformers | 5.19.0 |
| dtype | `torch.float32` |

> CPU 结果来自之前的记录，本次未在 A800 环境重跑 CPU 数据。

### 模型

| 项目 | 值 |
|---|---|
| 模型 | `/mnt/afs/models/Qwen/Qwen3-0.6B` |
| Prompt | `你好` |
| 输入 token 数 | 13 |
| 采样方式 | `temperature = 0`，greedy |
| `top_k` | `-1` |
| `top_p` | `1.0` |
| 生成长度 | 128 token，`--ignore-eos` |
| 正式运行 | 1 次 warmup + 3 次正式运行，取平均 |

## 测试方法

- 使用 `benchmark.py`
- GPU 计时前后调用 `torch.cuda.synchronize()`，保证异步 CUDA kernel 计时准确
- v0.1 / v0.2 使用历史 tag，均没有 KV cache，decode 每步重新计算完整序列
- 当前版本分别测试 `--no-use-kv-cache` 和 `--use-kv-cache`
- KV cache 模式下，prefill 后每步只喂入新 token；no-cache 模式下每步喂入完整序列

## GPU 结果：128 token

### v0.1

参数：`751,632,384`（未做 weight tying）

| Run | Total (s) | Generated | Prefill (ms) | Decode avg (ms/token) | Decode tok/s | Overall tok/s |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 3.434 | 128 | 26.8 | 26.74 | 37.40 | 37.28 |
| 2 | 3.655 | 128 | 26.5 | 28.48 | 35.12 | 35.02 |
| 3 | 3.427 | 128 | 26.7 | 26.69 | 37.47 | 37.35 |
| **平均** | **3.505** | **128** | **26.7** | **27.30** | **36.66** | **36.55** |

原始数据：[benchmark-cuda-v0.1.json](benchmark-cuda-v0.1.json)

### v0.2

参数：`596,049,920`（模块化实现 + weight tying，无 KV cache）

| Run | Total (s) | Generated | Prefill (ms) | Decode avg (ms/token) | Decode tok/s | Overall tok/s |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2.960 | 128 | 22.9 | 23.06 | 43.37 | 43.24 |
| 2 | 2.927 | 128 | 22.7 | 22.80 | 43.86 | 43.73 |
| 3 | 2.920 | 128 | 22.7 | 22.74 | 43.97 | 43.84 |
| **平均** | **2.936** | **128** | **22.8** | **22.87** | **43.73** | **43.61** |

原始数据：[benchmark-cuda-v0.2.json](benchmark-cuda-v0.2.json)

### 当前版本：no-cache

参数：`596,049,920`

| Run | Total (s) | Generated | Prefill (ms) | Decode avg (ms/token) | Decode tok/s | Overall tok/s |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2.992 | 128 | 23.3 | 23.31 | 42.90 | 42.77 |
| 2 | 2.993 | 128 | 23.2 | 23.32 | 42.89 | 42.77 |
| 3 | 2.987 | 128 | 23.3 | 23.27 | 42.98 | 42.86 |
| **平均** | **2.991** | **128** | **23.3** | **23.30** | **42.92** | **42.80** |

原始数据：[benchmark-cuda-current-nokv.json](benchmark-cuda-current-nokv.json)

### 当前版本：KV cache

参数：`596,049,920`

| Run | Total (s) | Generated | Prefill (ms) | Decode avg (ms/token) | Decode tok/s | Overall tok/s |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2.965 | 128 | 23.7 | 23.09 | 43.31 | 43.17 |
| 2 | 2.942 | 128 | 24.0 | 22.91 | 43.65 | 43.50 |
| 3 | 2.942 | 128 | 23.3 | 22.91 | 43.66 | 43.51 |
| **平均** | **2.950** | **128** | **23.7** | **22.97** | **43.54** | **43.39** |

原始数据：[benchmark-cuda-current-kv.json](benchmark-cuda-current-kv.json)

## 版本汇总

| 版本 | 参数量 | KV cache | Total (s) | Prefill (ms) | Decode avg (ms/token) | Decode tok/s | Overall tok/s |
|---|---:|---|---:|---:|---:|---:|---:|
| v0.1 | 751,632,384 | 否 | 3.505 | 26.7 | 27.30 | 36.66 | 36.55 |
| v0.2 | 596,049,920 | 否 | 2.936 | 22.8 | 22.87 | 43.73 | 43.61 |
| 当前 | 596,049,920 | 否 | 2.991 | 23.3 | 23.30 | 42.92 | 42.80 |
| 当前 | 596,049,920 | 是 | 2.950 | 23.7 | 22.97 | 43.54 | 43.39 |

观察：

- v0.1 → v0.2：总耗时从 `3.505 s` 降到 `2.936 s`，约 `-16.2%`；overall throughput 从 `36.55 tok/s` 提升到 `43.61 tok/s`，约 `+19.3%`
- v0.1 → 当前 KV cache：总耗时约 `-15.8%`，overall throughput 约 `+18.7%`
- v0.2 与当前 no-cache 基本持平，差异在运行波动范围内
- 当前 no-cache → 当前 KV cache：128 token 下差异约 `1.4%`，没有明显加速

## KV cache 长序列测试：512 token

为了检查长序列下缓存收益，额外对当前版本做了一次 512 token 测试：

| 模式 | Total (s) | Generated | Prefill (ms) | Decode avg (ms/token) | Decode tok/s | Overall tok/s |
|---|---:|---:|---:|---:|---:|---:|
| no-cache | 12.652 | 512 | 560.3 | 23.6 | 42.42 | 40.47 |
| KV cache | 13.042 | 512 | 571.8 | 24.3 | 41.14 | 39.26 |

原始数据：

- [benchmark-cuda-current-nokv-512.json](benchmark-cuda-current-nokv-512.json)
- [benchmark-cuda-current-kv-512.json](benchmark-cuda-current-kv-512.json)

结论：

- 在当前 `DynamicKVCache` 实现下，512 token 时 KV cache 没有带来加速，甚至略慢
- 主要原因是每层每步使用 `torch.cat` 动态拼接 K/V，且 batch=1 时 Python/launch 开销占主导
- 当前计算量虽然降低了，但缓存拼接、显存分配和 kernel launch 成本掩盖了收益
- 后续需要预分配 KV cache、使用 SDPA/FlashAttention、减少 Python 层循环或使用 CUDA Graphs，才能体现缓存收益

## 与旧 A100 数据对比

旧文档中的 A100 数据如下（128 token，`torch.bfloat16`）：

| 版本 | A100 overall tok/s | A800 overall tok/s | 变化 |
|---|---:|---:|---:|
| v0.1 | 31.67 | 36.55 | 约 +15.4% |
| v0.2 | 34.75 | 43.61 | 约 +25.5% |
| 当前 no-cache | 34.75 | 42.80 | 约 +23.2% |
| 当前 KV cache | 未测试 | 43.39 | — |

> 旧 A100 数据来自历史记录，当前 A800 使用项目相同 PyTorch / transformers 版本。不同机器的 CPU、驱动和功耗状态不同，以上对比仅作粗略参考。

## CPU 结果（历史数据）

### 8 token，3 次正式运行

| Run | Total (s) | Generated | Prefill (ms) | Decode avg (ms/token) | Decode tok/s | Overall tok/s |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2.450 | 8 | 262.3 | 312.0 | 3.20 | 3.27 |
| 2 | 3.822 | 8 | 263.8 | 507.8 | 1.97 | 2.09 |
| 3 | 2.420 | 8 | 228.8 | 312.5 | 3.20 | 3.31 |
| **平均** | **2.897** | **8** | **251.6** | **377.4** | **2.79** | **2.89** |

原始数据：[benchmark-cpu.json](benchmark-cpu.json)

### 32 token，1 次正式运行

| Run | Total (s) | Generated | Prefill (ms) | Decode avg (ms/token) | Decode tok/s | Overall tok/s |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 17.421 | 32 | 332.6 | 549.2 | 1.82 | 1.84 |

原始数据：[benchmark-cpu-32tok.json](benchmark-cpu-32tok.json)

## 复现命令

### v0.1

```bash
git worktree add /tmp/llm_infer_v01 v0.1
cd /tmp/llm_infer_v01

uv run python benchmark.py \
  --model /mnt/afs/models/Qwen/Qwen3-0.6B \
  --device cuda:0 \
  --max-new-tokens 128 \
  --warmup-runs 1 \
  --runs 3 \
  --ignore-eos \
  --json-out docs/benchmark-cuda-v0.1.json
```

### v0.2

```bash
git worktree add /tmp/llm_infer_v02 v0.2
cd /tmp/llm_infer_v02

uv run python benchmark.py \
  --model /mnt/afs/models/Qwen/Qwen3-0.6B \
  --device cuda:0 \
  --dtype bfloat16 \
  --max-new-tokens 128 \
  --warmup-runs 1 \
  --runs 3 \
  --ignore-eos \
  --json-out docs/benchmark-cuda-v0.2.json
```

### 当前版本：KV cache

```bash
cd /mnt/afs/zzh/code/LLM_INFER

uv run python benchmark.py \
  --model /mnt/afs/models/Qwen/Qwen3-0.6B \
  --device cuda:0 \
  --dtype bfloat16 \
  --max-new-tokens 128 \
  --warmup-runs 1 \
  --runs 3 \
  --ignore-eos \
  --use-kv-cache \
  --json-out docs/benchmark-cuda-current-kv.json
```

### 当前版本：no-cache

```bash
uv run python benchmark.py \
  --model /mnt/afs/models/Qwen/Qwen3-0.6B \
  --device cuda:0 \
  --dtype bfloat16 \
  --max-new-tokens 128 \
  --warmup-runs 1 \
  --runs 3 \
  --ignore-eos \
  --no-use-kv-cache \
  --json-out docs/benchmark-cuda-current-nokv.json
```

### 当前版本：512 token KV cache 对比

```bash
# no-cache
uv run python benchmark.py \
  --model /mnt/afs/models/Qwen/Qwen3-0.6B \
  --device cuda:0 \
  --dtype bfloat16 \
  --max-new-tokens 512 \
  --warmup-runs 0 \
  --runs 1 \
  --ignore-eos \
  --no-use-kv-cache \
  --json-out docs/benchmark-cuda-current-nokv-512.json

# KV cache
uv run python benchmark.py \
  --model /mnt/afs/models/Qwen/Qwen3-0.6B \
  --device cuda:0 \
  --dtype bfloat16 \
  --max-new-tokens 512 \
  --warmup-runs 0 \
  --runs 1 \
  --ignore-eos \
  --use-kv-cache \
  --json-out docs/benchmark-cuda-current-kv-512.json
```

也可以运行：

```bash
bash benchmark.sh
```

## 注意事项

- benchmark 结果会受 GPU 型号、驱动、PyTorch 版本、功耗状态和机器负载影响
- `temperature=0` 排除了采样随机性
- `--ignore-eos` 只用于固定长度性能测试，生成文本可能不自然
- 当前 KV cache 实现侧重正确性，尚未达到生产级性能
- 详细实现见 `src/python/engine/kv_cache.py` 和 `src/python/models/qwen3.py`
