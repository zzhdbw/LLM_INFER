# LLM_INFER

LLM_INFER 是一个基于 PyTorch 的轻量级 Qwen3 推理引擎。

模型结构从零实现，不直接使用 `transformers` 的模型实现；`transformers` 只用于读取配置、加载 Tokenizer 和解析 HuggingFace 模型元信息。项目采用模块化设计，包含基础层、Qwen3 模型、采样引擎、KV cache 和统一推理入口 `LLM`。

## 特性

- 手写 Qwen3 核心结构：
  - RMSNorm
  - RoPE（旋转位置编码）
  - GQA（Grouped Query Attention）
  - QK-Norm
  - SwiGLU MLP
  - Pre-Norm Transformer Decoder Layer
- 模块化组件：
  - `layers/`：基础算子和 Transformer 层
  - `models/`：Qwen3 模型、配置解析和权重加载
  - `engine/`：采样器 `Sampler` 和 `DynamicKVCache`
  - `llm/`：统一推理入口 `LLM`
- 支持从 HuggingFace 加载 Qwen3 配置和 Tokenizer
- 支持直接加载 safetensors 权重
- 支持 weight tying
- 支持 greedy、temperature、top-k、top-p 采样
- 支持 CPU / GPU、bfloat16 / float16 / float32
- 支持 KV cache，可通过 `--use-kv-cache` / `--no-use-kv-cache` 切换
- 提供 benchmark 脚本和 pre-commit 格式化配置

## 环境要求

- Python >= 3.11
- [uv](https://docs.astral.sh/uv/)
- PyTorch 2.8.0 + CUDA 12.8（CPU 也可以运行）
- transformers 5.19.0
- 模型：Qwen3-0.6B 或其他 Qwen3 模型

## 安装

```bash
cd /path/to/LLM_INFER
uv sync
```

`pyproject.toml` 已配置：

- PyPI 清华镜像
- PyTorch cu128 专用索引
- `torch` / `torchvision` / `torchaudio` 从 PyTorch 官方 CUDA 索引安装
- dev 依赖：`pre-commit`、`ruff`、`ty`

## Pre-commit

项目已配置 `.pre-commit-config.yaml`，提交前会自动执行：

1. `ruff check --fix`
2. `ruff format`

首次使用：

```bash
uv sync
uv run pre-commit install
```

手动运行：

```bash
uv run pre-commit run --all-files
```

## 快速开始

### 命令行

默认启用 KV cache：

```bash
uv run python main.py \
  --model /mnt/afs/models/Qwen/Qwen3-0.6B \
  --prompt "你好" \
  --device cuda:0
```

关闭 KV cache，每一步重新计算完整序列：

```bash
uv run python main.py \
  --model /mnt/afs/models/Qwen/Qwen3-0.6B \
  --prompt "你好" \
  --device cuda:0 \
  --no-use-kv-cache
```

也可以使用快捷脚本：

```bash
bash run.sh
```

使用 HuggingFace 模型名：

```bash
uv run python main.py \
  --model Qwen/Qwen3-0.6B \
  --prompt "你好" \
  --device cuda:0
```

### Python API

```python
import torch

from src.python import LLM, SamplingParams

llm = LLM(
    "/mnt/afs/models/Qwen/Qwen3-0.6B",
    dtype=torch.bfloat16,
    device="cuda:0",
)

results = llm.generate(
    ["你好"],
    SamplingParams(
        temperature=0.0,
        top_k=-1,
        top_p=1.0,
        max_tokens=32,
    ),
    use_kv_cache=True,
)

print(results[0]["text"])
print(results[0]["token_ids"])
```

关闭 KV cache：

```python
results = llm.generate(
    ["你好"],
    SamplingParams(max_tokens=32),
    use_kv_cache=False,
)
```

## 命令行参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--model` | 无，必填 | HuggingFace 模型名或本地模型目录 |
| `--prompt` | `你好` | 输入提示词 |
| `--max-tokens` | `128` | 最多生成 token 数 |
| `--temperature` | `0.7` | `0` 表示 greedy，值越大越随机 |
| `--top-k` | `50` | top-k 采样，`-1` 表示关闭 |
| `--top-p` | `0.9` | top-p 采样，`1.0` 表示关闭 |
| `--dtype` | `auto` | `auto`、`bfloat16`、`float16`、`float32` |
| `--device` | 自动选择 | `cuda:0`、`cuda:1`、`cpu` 等 |
| `--use-kv-cache` | `True` | 是否启用 KV cache；可用 `--no-use-kv-cache` 关闭 |

## KV cache 说明

每层缓存的 K/V 形状为：

```text
K_cache: [B, H_kv, S_cache, D]
V_cache: [B, H_kv, S_cache, D]
```

其中：

- `B`：batch size
- `H_kv`：KV head 数，GQA 时小于 query head 数
- `S_cache`：已缓存序列长度
- `D`：head dim

生成过程：

```text
Prefill:
  input_ids: [B, P]
  K/V cache: [B, H_kv, P, D]

Decode 第 1 步:
  input_ids: [B, 1]
  K/V cache: [B, H_kv, P + 1, D]

Decode 第 t 步:
  input_ids: [B, 1]
  K/V cache: [B, H_kv, P + t, D]
```

启用 KV cache 后，decode 阶段只计算新 token 的 Q/K/V、MLP 和 LM Head，历史 K/V 直接复用；关闭后每一步重新计算完整序列。

> 当前 `DynamicKVCache` 使用 `torch.cat` 动态拼接，实现简单、便于理解，但不是高性能实现。benchmark 中 128 / 512 token 场景下，KV cache 暂未体现明显加速，具体原因见 [docs/benchmark.md](docs/benchmark.md)。

## Benchmark

完整结果见 [docs/benchmark.md](docs/benchmark.md)。

A800 `cuda:0` 测试结果（Qwen3-0.6B，128 token，bfloat16，1 warmup + 3 runs）：

| 版本 | 参数量 | KV cache | Decode tok/s | Overall tok/s |
|---|---:|---|---:|---:|
| v0.1 | 751,632,384 | 否 | 36.66 | 36.55 |
| v0.2 | 596,049,920 | 否 | 43.73 | 43.61 |
| 当前 | 596,049,920 | 否 | 42.92 | 42.80 |
| 当前 | 596,049,920 | 是 | 43.54 | 43.39 |

- v0.1 → v0.2：overall throughput 提升约 `+19.3%`
- v0.1 → 当前 KV cache：overall throughput 提升约 `+18.7%`
- 当前 no-cache → 当前 KV cache：128 token 下差异约 `1.4%`
- 512 token 下，当前 cache 实现暂时略慢于 no-cache，原因主要是 `torch.cat`、Python 循环和 kernel launch 开销

GPU 测试环境为 NVIDIA A800-SXM4-80GB × 8，实际 benchmark 使用 `cuda:0`。

重新运行 benchmark：

```bash
# 当前版本 KV cache
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

# 当前版本 no-cache
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

v0.1 / v0.2 的复现命令见 [docs/benchmark.md](docs/benchmark.md)。

## 项目结构

```text
LLM_INFER/
├── main.py                       # 命令行推理入口
├── benchmark.py                  # benchmark 脚本
├── benchmark.sh                  # GPU benchmark 快捷脚本
├── run.sh                        # 快捷运行脚本
├── pyproject.toml                # 项目配置和依赖
├── uv.lock                       # 依赖锁定文件
├── .pre-commit-config.yaml       # pre-commit 配置
├── src/python/
│   ├── core.py                   # SamplingParams
│   ├── engine/
│   │   ├── sample.py             # Sampler
│   │   └── kv_cache.py           # DynamicKVCache
│   ├── layers/                   # 基础层和算子
│   │   ├── activation.py
│   │   ├── attention.py
│   │   ├── base.py
│   │   ├── embedding.py
│   │   ├── linear.py
│   │   ├── norm.py
│   │   └── rotary.py
│   ├── models/
│   │   ├── base.py               # BaseLLMModel
│   │   ├── config.py             # ModelConfig
│   │   ├── qwen3.py              # Qwen3 模型实现
│   │   └── weight.py             # safetensors 权重加载
│   └── llm/
│       └── llm.py                # LLM 推理入口
└── docs/
    ├── benchmark.md              # benchmark 结果与复现方法
    ├── benchmark-cuda-v0.1.json
    ├── benchmark-cuda-v0.2.json
    ├── benchmark-cuda-current-kv.json
    └── benchmark-cuda-current-nokv.json
```

## 实现说明

### 权重加载

`src/python/llm/llm.py` 中的 `LLM.__init__()` 会：

1. 解析 HuggingFace 模型名或本地模型目录
2. 用 `AutoConfig` 读取配置，转换成 `ModelConfig`
3. 在 `meta` 设备上创建模型结构，避免先分配真实显存
4. 用 `src/python/models/weight.py` 中的 `load_weights()` 读取 safetensors
5. 通过 `BaseOP.load_state_dict()` 将权重加载到模型
6. 将 RoPE 的 cos/sin cache 移动到目标设备

### 模型结构

`src/python/models/qwen3.py` 中实现了：

```text
Qwen3Attention
Qwen3MLP
Qwen3DecoderLayer
Qwen3Model
Qwen3ForCausalLM
```

其中 attention 使用 GQA + QK-Norm，MLP 使用 SwiGLU，Decoder 使用 Pre-Norm 残差结构。

### 生成过程

`src/python/llm/llm.py` 中的 `LLM.generate()` 是自回归生成：

1. 首次将完整 prompt 输入模型
2. 取最后一个位置的 logits
3. 使用 `Sampler` 根据 greedy / temperature / top-k / top-p 选出下一个 token
4. 拼回输入
5. 如果启用 KV cache，下一轮只输入新 token；否则每轮输入完整序列
6. 重复直到达到 `max_tokens` 或遇到 EOS

### KV cache

`src/python/models/qwen3.py` 中的 `Qwen3Attention` 在每层：

1. 计算当前 token 的 Q/K/V
2. 通过 `DynamicKVCache.update()` 将 K/V 追加到缓存
3. 使用完整缓存的 K/V 做 attention

`Qwen3Model.forward()` 会根据 `past_len` 生成正确的 `position_ids` 和 causal mask：

```text
position_ids = arange(past_len, past_len + S)
causal_mask = (S, past_len + S)
```

## 开发

```bash
# lint
uv run ruff check .

# format
uv run ruff format --check .

# pre-commit
uv run pre-commit run --all-files
```

## 已知限制

- 当前 `DynamicKVCache` 使用 `torch.cat` 动态拼接，长序列下拼接和 Python/launch 开销可能掩盖计算收益
- 没有预分配 KV cache、PagedAttention、continuous batching
- 当前只实现了 Qwen3 模型
- `LLM.generate()` 支持传入多个 prompt，但内部逐条生成，没有 batch 并行
- benchmark 结果依赖硬件、PyTorch 版本、线程数、驱动和功耗状态

## License

见 [LICENSE](LICENSE)。
