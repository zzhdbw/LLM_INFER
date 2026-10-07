# LLM_INFER

LLM_INFER 是一个基于 PyTorch 的轻量级 Qwen3 推理引擎。

模型结构从零实现，不直接使用 `transformers` 的模型实现；`transformers` 只用于读取配置、加载 Tokenizer 和解析 HuggingFace 模型元信息。项目采用模块化设计，包含层实现、模型实现、采样引擎和统一推理入口 `LLM`。

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
  - `engine/`：采样器 `Sampler`
  - `llm/`：统一推理入口 `LLM`
- 支持从 HuggingFace 加载 Qwen3 配置和 Tokenizer
- 支持直接加载 safetensors 权重
- 支持 weight tying
- 支持 greedy、temperature、top-k、top-p 采样
- 支持 CPU / GPU、bfloat16 / float16 / float32
- 提供 benchmark 脚本和 pre-commit 格式化配置
- 当前为无 KV cache 的教学实现，便于理解 decode 阶段计算量

## 环境要求

- Python >= 3.11
- [uv](https://docs.astral.sh/uv/)
- PyTorch 2.8.0 + CUDA 12.8（CPU 也可以运行）
- transformers 5.19.0
- 模型：Qwen3-0.6B 或其他 Qwen3 模型

## 安装

```bash
cd ~/code/INFRA/LLM_INFER
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

```bash
uv run python main.py \
  --model /home/models/Qwen/Qwen3-0.6B \
  --prompt "你好" \
  --device cuda:0
```

或：

```bash
bash run.sh
```

HuggingFace 模型名：

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
    "/home/models/Qwen/Qwen3-0.6B",
    dtype=torch.bfloat16,
    device="cuda:0",
)

results = llm.generate(
    ["你好"],
    SamplingParams(
        temperature=0.0,
        max_tokens=32,
    ),
)

print(results[0]["text"])
print(results[0]["token_ids"])
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
│   │   └── sample.py             # Sampler
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
    └── benchmark.md              # benchmark 结果与复现方法
```

## Benchmark

Benchmark 结果详见 [docs/benchmark.md](docs/benchmark.md)。

| 设备 | 生成长度 | dtype | Decode tok/s | Overall tok/s |
|---|---:|---|---:|---:|
| CPU | 8 | `torch.float32` | 2.79 | 2.89 |
| CPU | 32 | `torch.float32` | 1.82 | 1.84 |
| A100 `cuda:0` | 128 | `torch.bfloat16` | 34.85 | 34.75 |

GPU 测试环境为 NVIDIA A100-SXM4-80GB，实际 benchmark 使用 `cuda:0`。

重新运行：

```bash
# CPU 8 token，3 次取平均
uv run python benchmark.py \
  --model /home/models/Qwen/Qwen3-0.6B \
  --device cpu \
  --dtype float32 \
  --max-new-tokens 8 \
  --warmup-runs 1 \
  --runs 3 \
  --json-out docs/benchmark-cpu.json

# GPU 128 token
bash benchmark.sh
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

1. 将完整序列输入模型
2. 取最后一个位置的 logits
3. 使用 `Sampler` 根据 greedy / temperature / top-k / top-p 选出下一个 token
4. 拼回输入，重复直到达到 `max_tokens` 或遇到 EOS

当前实现没有 KV cache，因此每一步都会重新计算整个序列，decode 复杂度较高。后续可以加入 KV cache 优化。

## 已知限制

- 无 KV cache，序列越长 decode 越慢
- 当前只实现了 Qwen3 模型
- `LLM.generate()` 支持传入多个 prompt，但内部逐条生成，没有 batch 并行和 continuous batching
- benchmark 结果依赖硬件、PyTorch 版本、线程数和模型精度

## License

见 [LICENSE](LICENSE)。
