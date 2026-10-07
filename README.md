# LLM_INFER

从零手写 Qwen3 的端到端推理项目，适合学习大模型推理原理。

项目只依赖 PyTorch 实现 Qwen3 的模型结构，不直接使用 `transformers` 的模型实现，只使用 `transformers` 从 HuggingFace 加载配置、Tokenizer 和权重，然后完成自回归文本生成。

## 特性

- 手写 Qwen3 核心结构：
  - RMSNorm
  - RoPE（旋转位置编码）
  - GQA（Grouped Query Attention）
  - QK-Norm
  - SwiGLU MLP
  - Pre-Norm Transformer Decoder Layer
- 支持从 HuggingFace 加载 Qwen3 权重
- 支持 greedy、temperature、top-k、top-p 采样
- 提供 benchmark 脚本，方便对比 CPU / GPU 推理性能
- 无 KV cache 的教学实现，便于理解 decode 阶段的计算量

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

## 运行

使用本地模型：

```bash
uv run python main.py \
  --model /home/models/Qwen/Qwen3-0.6B \
  --prompt "你好" \
  --device cuda:0
```

或者直接：

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

### 参数说明

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--model` | 无，必填 | HuggingFace 模型名或本地路径 |
| `--prompt` | `What is the capital of France?` | 输入提示词 |
| `--max-tokens` | `128` | 最多生成 token 数 |
| `--temperature` | `0.7` | `0` 表示 greedy，值越大越随机 |
| `--top-k` | `50` | top-k 采样 |
| `--top-p` | `0.9` | top-p 采样 |
| `--device` | 自动选择 | `cuda:0`、`cuda:1`、`cpu` 等 |

## 项目结构

```text
LLM_INFER/
├── main.py                    # Qwen3 手写模型 + 推理入口
├── benchmark.py               # 推理 benchmark 脚本
├── benchmark.sh               # GPU benchmark 快捷脚本
├── run.sh                     # 快捷运行脚本
├── pyproject.toml             # uv 项目配置和依赖
├── uv.lock                    # 依赖锁定文件
└── docs/
    └── benchmark.md           # benchmark 结果与复现方法
```

## Benchmark

Benchmark 结果详见 [docs/benchmark.md](docs/benchmark.md)，当前已包含 CPU 和 GPU 结果。

| 设备 | 生成长度 | dtype | Decode tok/s | Overall tok/s |
|---|---:|---|---:|---:|
| CPU | 8 | `torch.float32` | 1.98 | 2.01 |
| CPU | 32 | `torch.float32` | 1.44 | 1.45 |
| A100 `cuda:0` | 128 | `torch.bfloat16` | 31.76 | 31.67 |

GPU 测试环境为 NVIDIA A100-SXM4-80GB，实际 benchmark 使用 `cuda:0`。

运行方式：

```bash
# CPU 8 token，3 次取平均
uv run python benchmark.py \
  --model /home/models/Qwen/Qwen3-0.6B \
  --device cpu \
  --max-new-tokens 8 \
  --warmup-runs 1 \
  --runs 3 \
  --json-out docs/benchmark-cpu.json

# GPU 长序列 benchmark，例如 128 token
uv run python benchmark.py \
  --model /home/models/Qwen/Qwen3-0.6B \
  --device cuda:0 \
  --max-new-tokens 128 \
  --warmup-runs 1 \
  --runs 3 \
  --ignore-eos \
  --json-out docs/benchmark-cuda.json
```

也可以直接运行：

```bash
bash benchmark.sh
```

## 实现说明

### 权重加载

`main.py` 中的 `load_weights_from_hf()` 会：

1. 用 HuggingFace `AutoModelForCausalLM.from_pretrained()` 加载官方模型
2. 取出 `state_dict`
3. 加载到手写的 `Qwen3ForCausalLM`
4. 删除 HuggingFace 模型并释放显存

这样可以避免手写 RoPE、GQA、权重分片等一堆加载逻辑。

### 生成过程

`generate()` 是自回归生成：

1. 将完整序列输入模型
2. 取最后一个位置的 logits
3. 根据 greedy / temperature / top-k / top-p 选出下一个 token
4. 拼回输入，重复直到达到 `max_tokens` 或遇到 EOS

当前实现没有 KV cache，因此每一步都会重新计算整个序列，decode 复杂度较高。后续可以加入 KV cache 优化。

## 已知限制

- 无 KV cache，序列越长 decode 越慢
- benchmark 结果依赖硬件、PyTorch 版本、线程数和模型精度
- 当前手写实现没有做 `lm_head` 和 `embed_tokens` 的 weight tying，因此 Qwen3-0.6B 的参数量显示为约 `0.75B`
- `load_weights_from_safetensors` 目前要求本地 safetensors 目录，主流程使用 `load_weights_from_hf`

## License

见 [LICENSE](LICENSE)。
