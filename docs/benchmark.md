# Benchmark

本文记录 Qwen3 手写推理实现 v0.1、v0.2 和当前版本在 **NVIDIA A800** 上的 benchmark 结果。

> v0.1 和 v0.2 使用历史 tag 运行
>
> 当前版本包含预分配 KV cache 实现
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
| 生成长度 | 128 / 512 / 1024 token，`--ignore-eos` |
| 正式运行 | 128 token：1 warmup + 3 runs；512 token：1 warmup + 2 runs；1024 token：0 warmup + 1 run |

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
| 1 | 2.908 | 128 | 22.5 | 22.66 | 44.13 | 44.01 |
| 2 | 2.903 | 128 | 22.5 | 22.62 | 44.21 | 44.09 |
| 3 | 2.899 | 128 | 22.6 | 22.59 | 44.27 | 44.15 |
| **平均** | **2.904** | **128** | **22.5** | **22.62** | **44.21** | **44.08** |

原始数据：[benchmark-cuda-current-nokv.json](benchmark-cuda-current-nokv.json)

### 当前版本：KV cache（Preallocated）

参数：`596,049,920`

| Run | Total (s) | Generated | Prefill (ms) | Decode avg (ms/token) | Decode tok/s | Overall tok/s |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2.991 | 128 | 24.5 | 23.29 | 42.93 | 42.79 |
| 2 | 2.983 | 128 | 24.5 | 23.23 | 43.05 | 42.91 |
| 3 | 2.990 | 128 | 24.4 | 23.28 | 42.95 | 42.82 |
| **平均** | **2.988** | **128** | **24.5** | **23.27** | **42.98** | **42.84** |

原始数据：[benchmark-cuda-current-kv.json](benchmark-cuda-current-kv.json)

## 版本汇总

| 版本 | 参数量 | KV cache | Total (s) | Prefill (ms) | Decode avg (ms/token) | Decode tok/s | Overall tok/s |
|---|---:|---|---:|---:|---:|---:|---:|
| v0.1 | 751,632,384 | 否 | 3.505 | 26.7 | 27.30 | 36.66 | 36.55 |
| v0.2 | 596,049,920 | 否 | 2.936 | 22.8 | 22.87 | 43.73 | 43.61 |
| 当前 | 596,049,920 | 否 | 2.904 | 22.5 | 22.62 | 44.21 | 44.08 |
| 当前 | 596,049,920 | 是 | 2.988 | 24.5 | 23.27 | 42.98 | 42.84 |

观察：

- v0.1 → v0.2：总耗时从 `3.505 s` 降到 `2.936 s`，约 `-16.2%`；overall throughput 从 `36.55 tok/s` 提升到 `43.61 tok/s`，约 `+19.3%`
- v0.1 → 当前 KV cache：总耗时约 `-14.8%`，overall throughput 约 `+17.2%`
- v0.2 与当前 no-cache 基本持平，差异在运行波动范围内
- 当前 no-cache → 当前 KV cache：128 token 下 cache 略慢约 2.8%；512 token 下基本持平；1024 token 下 cache 明显更快

## KV cache 长序列测试：512 token

为了检查长序列下缓存收益，额外对当前版本做了一次 512 token 测试：

| 模式 | Total (s) | Generated | Prefill (ms) | Decode avg (ms/token) | Decode tok/s | Overall tok/s |
|---|---:|---:|---:|---:|---:|---:|
| no-cache | 12.122 | 512 | 23.6 | 23.61 | 42.36 | 42.24 |
| KV cache（Preallocated） | 12.018 | 512 | 25.9 | 23.40 | 42.74 | 42.60 |

原始数据：

- [benchmark-cuda-current-nokv-512.json](benchmark-cuda-current-nokv-512.json)
- [benchmark-cuda-current-kv-512.json](benchmark-cuda-current-kv-512.json)

> `prefill` 是首步耗时，包含首次 CUDA 预热和缓存分配，波动较大；decode avg 更能反映稳定生成阶段。

结论：

- 512 token 下，cache 与 no-cache 基本持平
- decode 分别为 `23.61 ms/token` 和 `23.40 ms/token`，差异小于 `1%`
- 短序列下 Python / kernel launch 开销占主导，KV cache 的收益不明显
- 预分配 cache 已经消除了每步 `torch.cat`，但需要更长序列或更大 batch 才能体现收益

## KV cache 长序列测试：1024 token

继续增加生成长度到 1024 token：

| 模式 | Total (s) | Generated | Prefill (ms) | Decode avg (ms/token) | Decode tok/s | Overall tok/s |
|---|---:|---:|---:|---:|---:|---:|
| no-cache | 63.574 | 1024 | 2444.8 | 59.67 | 16.76 | 16.11 |
| KV cache（Preallocated） | 26.445 | 1024 | 2538.9 | 23.29 | 42.93 | 38.72 |

原始数据：

- [benchmark-cuda-current-nokv-1024.json](benchmark-cuda-current-nokv-1024.json)
- [benchmark-cuda-current-kv-1024.json](benchmark-cuda-current-kv-1024.json)

结论：

- 1024 token 下 KV cache 优势开始非常明显
- decode 单 token 耗时从 `59.67 ms` 降到 `23.29 ms`，吞吐从 `16.76 tok/s` 提升到 `42.93 tok/s`，约 `2.56x`
- overall throughput 从 `16.11 tok/s` 提升到 `38.72 tok/s`，约 `2.40x`
- no-cache 在长序列下需要反复计算完整序列，attention 和 MLP 成本快速增长
- KV cache 模式下 dense 部分只计算新 token，因此 decode 耗时基本保持稳定

## 与旧 A100 数据对比

旧文档中的 A100 数据如下（128 token，`torch.bfloat16`）：

| 版本 | A100 overall tok/s | A800 overall tok/s | 变化 |
|---|---:|---:|---:|
| v0.1 | 31.67 | 36.55 | 约 +15.4% |
| v0.2 | 34.75 | 43.61 | 约 +25.5% |
| 当前 no-cache | 34.75 | 44.08 | 约 +26.8% |
| 当前 KV cache | 未测试 | 42.84 | — |

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

### 当前版本：KV cache（Preallocated）

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
  --warmup-runs 1 \
  --runs 2 \
  --ignore-eos \
  --no-use-kv-cache \
  --json-out docs/benchmark-cuda-current-nokv-512.json

# KV cache
uv run python benchmark.py \
  --model /mnt/afs/models/Qwen/Qwen3-0.6B \
  --device cuda:0 \
  --dtype bfloat16 \
  --max-new-tokens 512 \
  --warmup-runs 1 \
  --runs 2 \
  --ignore-eos \
  --use-kv-cache \
  --json-out docs/benchmark-cuda-current-kv-512.json
```

### 当前版本：1024 token KV cache 对比

```bash
# no-cache
uv run python benchmark.py \
  --model /mnt/afs/models/Qwen/Qwen3-0.6B \
  --device cuda:0 \
  --dtype bfloat16 \
  --max-new-tokens 1024 \
  --warmup-runs 0 \
  --runs 1 \
  --ignore-eos \
  --no-use-kv-cache \
  --json-out docs/benchmark-cuda-current-nokv-1024.json

# KV cache
uv run python benchmark.py \
  --model /mnt/afs/models/Qwen/Qwen3-0.6B \
  --device cuda:0 \
  --dtype bfloat16 \
  --max-new-tokens 1024 \
  --warmup-runs 0 \
  --runs 1 \
  --ignore-eos \
  --use-kv-cache \
  --json-out docs/benchmark-cuda-current-kv-1024.json
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
