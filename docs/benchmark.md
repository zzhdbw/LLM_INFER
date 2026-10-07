# Benchmark

本文记录当前模块化 Qwen3 推理实现的 benchmark 结果。

> 说明：当前提供 **最新版 CPU 和 GPU benchmark**。所有结果均基于模块化实现。

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
| GPU | NVIDIA A100-SXM4-80GB（CUDA 设备 0） |
| GPU 显存 | 81920 MiB / 80 GB |
| 状态 | 已使用当前模块化实现测试 |

### 模型

| 项目 | 值 |
|---|---|
| 模型 | `/home/models/Qwen/Qwen3-0.6B` |
| 手写模型参数量 | 596,049,920（0.60B） |
| Prompt | `你好` |
| 输入 token 数 | 13 |
| 采样方式 | `temperature = 0`，greedy |
| `top_k` | `-1` |
| `top_p` | `1.0` |

## 测试方法

- 使用 `benchmark.py` 调用 `src.python.LLM` 和 `Sampler`
- 仅计时生成阶段，不包括模型加载和权重加载时间
- CPU 结果使用 `torch.float32`
- 8 token：1 次 warmup + 3 次正式运行，取平均
- 32 token：1 次正式运行，使用 `--ignore-eos` 强制生成满 32 个 token
- GPU 运行时会使用 `torch.cuda.synchronize()` 保证异步 CUDA kernel 计时准确

当前实现没有 KV cache，decode 每一步都会重新计算完整序列，因此长序列下单 token 耗时会更长。

## CPU 结果：8 token，3 次正式运行

| Run | Total (s) | Generated | Prefill (ms) | Decode avg (ms/token) | Decode tok/s | Overall tok/s |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2.450 | 8 | 262.3 | 312.0 | 3.20 | 3.27 |
| 2 | 3.822 | 8 | 263.8 | 507.8 | 1.97 | 2.09 |
| 3 | 2.420 | 8 | 228.8 | 312.5 | 3.20 | 3.31 |
| **平均** | **2.897** | **8** | **251.6** | **377.4** | **2.79** | **2.89** |

输出示例：

```text
你好！有什么可以帮助你的吗？
```

原始数据：[benchmark-cpu.json](benchmark-cpu.json)

## CPU 结果：32 token，1 次正式运行

| Run | Total (s) | Generated | Prefill (ms) | Decode avg (ms/token) | Decode tok/s | Overall tok/s |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 17.421 | 32 | 332.6 | 549.2 | 1.82 | 1.84 |

原始数据：[benchmark-cpu-32tok.json](benchmark-cpu-32tok.json)

## GPU 结果：128 token，3 次正式运行

| Run | Total (s) | Generated | Prefill (ms) | Decode avg (ms/token) | Decode tok/s | Overall tok/s |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 3.724 | 128 | 28.3 | 29.01 | 34.47 | 34.37 |
| 2 | 3.655 | 128 | 28.6 | 28.48 | 35.11 | 35.02 |
| 3 | 3.672 | 128 | 28.4 | 28.61 | 34.95 | 34.86 |
| **平均** | **3.684** | **128** | **28.5** | **28.70** | **34.85** | **34.75** |

原始数据：[benchmark-cuda.json](benchmark-cuda.json)

## 结果汇总

| 设备 | 生成长度 | dtype | Decode avg | Decode tok/s | Overall tok/s |
|---|---:|---|---:|---:|---:|
| CPU | 8 | `torch.float32` | 377.4 ms/token | 2.79 | 2.89 |
| CPU | 32 | `torch.float32` | 549.2 ms/token | 1.82 | 1.84 |
| A100 `cuda:0` | 128 | `torch.bfloat16` | 28.70 ms/token | 34.85 | 34.75 |

观察：

- 无 KV cache 时，CPU 上序列从 8 token 增长到 32 token，decode 平均耗时从 `377.4 ms/token` 增加到 `549.2 ms/token`，约增加 `45.5%`
- GPU A100 在 128 token 时 decode 平均 `28.70 ms/token`，整体 `34.75 tok/s`
- CPU 与 GPU 测试长度和 dtype 不完全相同，但性能差距在一个数量级以上
- 当前模块化实现使用了 weight tying，Qwen3-0.6B 参数量为 `596,049,920`
- CPU 使用 `float32`，GPU 使用 `bfloat16`，两者不是严格等价对比

## 复现命令

### CPU 8 token

```bash
uv run python benchmark.py \
  --model /home/models/Qwen/Qwen3-0.6B \
  --device cpu \
  --dtype float32 \
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
  --dtype float32 \
  --max-new-tokens 32 \
  --warmup-runs 0 \
  --runs 1 \
  --ignore-eos \
  --json-out docs/benchmark-cpu-32tok.json
```

### GPU 128 token

```bash
bash benchmark.sh
```

等价手动命令：

```bash
uv run python benchmark.py \
  --model /home/models/Qwen/Qwen3-0.6B \
  --device cuda:0 \
  --dtype bfloat16 \
  --max-new-tokens 128 \
  --warmup-runs 1 \
  --runs 3 \
  --ignore-eos \
  --json-out docs/benchmark-cuda.json
```

## 注意事项

- benchmark 结果会受 CPU/GPU 型号、PyTorch 版本、线程数、功耗模式等影响
- `temperature=0` 排除了采样随机性，但换成采样解码后速度会略有变化
- `--ignore-eos` 只推荐用于固定长度性能测试，生成文本可能不自然
- 最新 GPU 数据已通过 `bash benchmark.sh` 使用当前模块化实现生成
