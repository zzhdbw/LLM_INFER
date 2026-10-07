# Benchmark

本文记录当前手写 Qwen3 推理实现的 benchmark 结果，包括 CPU baseline 和 GPU 结果。

## 测试环境

### CPU

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

### GPU

| 项目 | 值 |
|---|---|
| GPU | NVIDIA A100-SXM4-80GB × 8（benchmark 使用 `cuda:0`） |
| GPU 显存 | 81920 MiB / 80 GB |
| 操作系统 | Linux 5.15.0-140-generic x86_64 |
| Python | 3.11.11 |
| PyTorch | 2.8.0+cu128 |
| CUDA | 启用，`cuda:0` |
| transformers | 5.19.0 |
| dtype | `torch.bfloat16` |

### 模型

| 项目 | 值 |
|---|---|
| 模型 | `/home/models/Qwen/Qwen3-0.6B` |
| 手写模型参数量 | 751,632,384 |
| Prompt | `你好` |
| 输入 token 数 | 13 |
| 采样方式 | `temperature = 0`，即 greedy 解码 |

## 测试方法

- 使用 `benchmark.py` 调用 `main.py` 中的 `generate()`
- 仅计时生成阶段，不包括模型加载和权重加载时间
- GPU 每次计时前后会 `torch.cuda.synchronize()`，保证异步 CUDA kernel 计时准确
- CPU 结果：
  - 8 token：1 次 warmup + 3 次正式运行，取平均
  - 32 token：1 次正式运行，使用 `--ignore-eos` 强制生成满 32 个 token
- GPU 结果：
  - 128 token：1 次 warmup + 3 次正式运行，取平均
  - 使用 `--ignore-eos` 强制生成满 128 个 token

由于当前实现没有 KV cache，decode 每一步都会重新计算完整序列，所以长序列下单 token 耗时会更长。

## GPU 结果：128 token，3 次正式运行

| Run | Total (s) | Generated | Prefill (ms) | Decode avg (ms/token) | Decode tok/s | Overall tok/s |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4.048 | 128 | 31.4 | 31.54 | 31.71 | 31.62 |
| 2 | 4.044 | 128 | 31.5 | 31.50 | 31.75 | 31.65 |
| 3 | 4.034 | 128 | 31.2 | 31.43 | 31.82 | 31.73 |
| **平均** | **4.042** | **128** | **31.4** | **31.49** | **31.76** | **31.67** |

原始数据：[benchmark-cuda.json](benchmark-cuda.json)

## CPU 结果：8 token，3 次正式运行

| Run | Total (s) | Generated | Prefill (ms) | Decode avg (ms/token) | Decode tok/s | Overall tok/s |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4.571 | 8 | 480.6 | 583.9 | 1.71 | 1.75 |
| 2 | 4.073 | 8 | 489.3 | 511.5 | 1.96 | 1.96 |
| 3 | 3.468 | 8 | 389.7 | 439.3 | 2.28 | 2.31 |
| **平均** | **4.038** | **8** | **453.2** | **511.6** | **1.98** | **2.01** |

输出示例：

```text
你好！有什么可以帮助你的吗？
```

原始数据：[benchmark-cpu.json](benchmark-cpu.json)

## CPU 结果：32 token，1 次正式运行

| Run | Total (s) | Generated | Prefill (ms) | Decode avg (ms/token) | Decode tok/s | Overall tok/s |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 22.033 | 32 | 466.0 | 693.8 | 1.44 | 1.45 |

原始数据：[benchmark-cpu-32tok.json](benchmark-cpu-32tok.json)

## CPU / GPU 对比

| 设备 | 生成长度 | dtype | Decode avg | Decode tok/s | Overall tok/s |
|---|---:|---|---:|---:|---:|
| CPU | 8 | `torch.float32` | 511.6 ms/token | 1.98 | 2.01 |
| CPU | 32 | `torch.float32` | 693.8 ms/token | 1.44 | 1.45 |
| A100 `cuda:0` | 128 | `torch.bfloat16` | 31.49 ms/token | 31.76 | 31.67 |

说明：

- GPU 使用 `bfloat16`，CPU 使用 `float32`，数值并不严格等价
- CPU 32 token 与 GPU 128 token 的序列长度也不完全相同
- 但数量级差异非常明显：GPU decode 约为 CPU 的 20 倍以上
- CPU 无 KV cache，在 8 → 32 token 时，decode 单 token 耗时从 `511.6 ms` 增加到 `693.8 ms`，约增加 `35.6%`
- GPU 无 KV cache，在 128 token 时仍能保持约 `31.76 tok/s` 的 decode 速度

## 复现命令

### GPU 128 token

```bash
uv run python benchmark.py \
  --model /home/models/Qwen/Qwen3-0.6B \
  --device cuda:0 \
  --max-new-tokens 128 \
  --warmup-runs 1 \
  --runs 3 \
  --ignore-eos \
  --json-out docs/benchmark-cuda.json
```

### CPU 8 token

```bash
uv run python benchmark.py \
  --model /home/models/Qwen/Qwen3-0.6B \
  --device cpu \
  --max-new-tokens 8 \
  --warmup-runs 1 \
  --runs 3 \
  --json-out docs/benchmark-cpu.json
```

### CPU 32 token

```bash
uv run python benchmark.py \
  --model /home/models/Qwen/Qwen3-0.6B \
  --device cpu \
  --max-new-tokens 32 \
  --warmup-runs 0 \
  --runs 1 \
  --ignore-eos \
  --json-out docs/benchmark-cpu-32tok.json
```

## 注意事项

- benchmark 结果会受 CPU/GPU 型号、PyTorch 版本、线程数、功耗模式等影响
- `temperature=0` 排除了采样随机性，但如果换成采样解码，速度会略有变化
- `--ignore-eos` 只推荐用于固定长度的性能测试，生成的文本可能不自然
