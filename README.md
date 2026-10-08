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
  - `engine/`：采样器 `Sampler`、`DynamicKVCache` 和 `PreallocatedKVCache`
  - `llm/`：统一推理入口 `LLM`
- 支持从 HuggingFace 加载 Qwen3 配置和 Tokenizer
- 支持直接加载 safetensors 权重
- 支持 weight tying
- 支持 greedy、temperature、top-k、top-p 采样
- 支持 CPU / GPU、bfloat16 / float16 / float32
- 支持 KV cache：默认使用预分配 `PreallocatedKVCache`，最大序列长度可通过 `--max-seq-len` 指定
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
| `--max-seq-len` | `4096` | 预分配 KV cache 的最大序列长度，需 `>= prompt_len + max_tokens` |
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

当前默认使用 `PreallocatedKVCache`：

- 初始化时一次性分配 `(batch, num_kv_heads, max_seq_len, head_dim)` 的 K/V
- decode 时按当前位置原地写入，不再每步执行 `torch.cat`
- `max_seq_len` 可通过 `LLM(..., max_seq_len=...)` 或命令行 `--max-seq-len` 指定
- 如果实际长度超过 `max_seq_len`，会直接报错，提示增大 `max_seq_len` 或减小 `--max-tokens`

旧版 `DynamicKVCache` 仍保留在 `engine/kv_cache.py`，用于对比和教学。长序列下预分配版本通常能获得更明显的收益，具体数据见 [docs/benchmark.md](docs/benchmark.md)。

## Benchmark

完整结果见 [docs/benchmark.md](docs/benchmark.md)。

A800 `cuda:0` 测试结果（Qwen3-0.6B，128 token，bfloat16，1 warmup + 3 runs）：

| 版本 | 参数量 | KV cache | Decode tok/s | Overall tok/s |
|---|---:|---|---:|---:|
| v0.1 | 751,632,384 | 否 | 36.66 | 36.55 |
| v0.2 | 596,049,920 | 否 | 43.73 | 43.61 |
| 当前 | 596,049,920 | 否 | 44.21 | 44.08 |
| 当前 | 596,049,920 | 是 | 42.98 | 42.84 |

- v0.1 → v0.2：overall throughput 提升约 `+19.3%`
- v0.1 → 当前 KV cache：overall throughput 提升约 `+17.2%`

### KV cache 随序列长度的收益

| 生成长度 | no-cache Decode tok/s | KV cache Decode tok/s | 变化 |
|---:|---:|---:|---:|
| 128 token | 44.21 | 42.98 | 约 `-2.8%` |
| 512 token | 42.36 | 42.74 | 约 `+0.9%` |
| 1024 token | 16.76 | 42.93 | 约 `+156%`（`2.56x`） |

结论：

- 128 token：序列太短，Python / kernel launch 开销占主导，cache 反而略慢
- 512 token：cache 与 no-cache 基本持平
- 1024 token：cache 明显领先，因为 no-cache 每一步都要重新计算完整序列
- `PreallocatedKVCache` 避免了每步 `torch.cat`，但优势需要足够长的序列才能体现

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

# 长序列 KV cache 对比：1024 token
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
│   │   └── kv_cache.py           # DynamicKVCache / PreallocatedKVCache
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
2. 通过 `PreallocatedKVCache.update()` 将 K/V 原地写入预分配缓存
3. 使用完整有效前缀 K/V 做 attention

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

- 当前 KV cache 仍是单请求、batch=1 的实现，尚未使用 PagedAttention / continuous batching / CUDA Graphs；Python 层循环和 kernel launch 开销仍然存在
- 当前只实现了 Qwen3 模型
- `LLM.generate()` 支持传入多个 prompt，但内部逐条生成，没有 batch 并行
- benchmark 结果依赖硬件、PyTorch 版本、线程数、驱动和功耗状态

## License

见 [LICENSE](LICENSE)。
