# CLAUDE.md

## Project Overview

tinychat-mlx is a self-contained MLX training framework for Apple Silicon. Full pipeline from data download to chat, no PyTorch required. Includes general tool calling: register any Python function and the model calls it by name.

## Setup

```bash
uv sync                       # Install dependencies (use uv, not pip)
source .venv/bin/activate
```

Python 3.10+.

## Tests

Always test with `make`. See [TESTING.md](TESTING.md) for full details.

```bash
make test            # All tests
make test-tools      # Tool calling tests only
make test-fast       # Skip slow tests
make test-gpt        # GPT model tests only
```

## Common Commands

```bash
# Quick start (web GUI at http://127.0.0.1:8000)
python -m scripts.quickstart

# Full pipeline via Make
make train-d4        # data + tokenizer + pretrain + SFT at depth=4
make train-d12       # same at depth=12

# Individual steps
make data            # download 8 FineWeb-edu shards
make tokenizer       # train BPE tokenizer
make pretrain-d4     # pretrain base model at depth=4
make sft-d4          # SFT at depth=4 (includes tool calling data)
make chat-d4         # interactive chat (depth=4, sft checkpoint)
make eval-d4         # run benchmarks

# Preview tool calling training data
make preview-tools

# Import pretrained model (requires uv sync --extra convert)
python -m scripts.convert_from_hf --repo your-hf-org/your-model
```

## SFT Tool Calling Flags

```bash
python -m scripts.sft --depth=12 --tool-epochs=2    # default: 2 epochs of 50k tool examples
python -m scripts.chat --depth=12 --source=sft --no-tools   # disable tool registry
```

## Architecture

Single complexity dial: `--depth` controls everything. Width, heads, batch size, LR, training tokens all auto-computed. Valid range: 4–26. Any change must work across all depths ("miniseries principle").

### Key Modules (tinychat_mlx/)

- **gpt.py**: GPT transformer (RoPE, QK-norm, ReLU², GQA, sliding window, logit softcap, value embeddings, per-layer residual scaling)
- **optim.py**: Muon (matrix params) + AdamW (scalar params) multi-optimizer
- **engine.py**: Inference with KV cache; `Engine(model, tokenizer, tool_registry=None)` — dispatches `<|python_start|>tool_name(kwarg=val)<|python_end|>` to `ToolRegistry`, falls back to calculator
- **tools.py**: `ToolRegistry` — `@registry.register(description=...)` decorator, `dispatch(expr)` via `ast.literal_eval` per kwarg (never `eval()`), `system_prompt_block()` for system prompt injection
- **train.py**: Training loop with gradient accumulation, checkpointing, resume
- **sft.py**: SFT pipeline; training mixture includes `ToolCallingTask`
- **dataloader.py**: BOS-aligned best-fit packing
- **tokenizer.py**: BPE tokenizer (RustBPE + tiktoken); `render_conversation()` handles `python`/`python_output` part types and system message merging

### Tool Calling Format

The model generates Python-style function calls using existing special tokens (no vocab change needed):

```
<|python_start|>tool_name(arg="value")<|python_end|>
<|output_start|>result<|output_end|>
```

Training data (`tasks/tool_calling.py`) produces conversations with assistant content as `[{"type": "text"}, {"type": "python"}, {"type": "python_output"}, ...]` parts — the exact format `render_conversation()` already handles. Mask=1 on `python` parts (supervised), mask=0 on `python_output` (injected at runtime).

### Key MLX Patterns

- Call `mx.eval()` after every micro-batch — prevents computation graph buildup
- MLX parameters are lazy by default (no meta device needed)
- Weight loading via `_load_weights_into_model()` using getattr/setattr tree walk
- Memory capped with `metal.set_memory_limit(gb)` in `common.py`

## Artifacts

Stored under `weights/` at the project root (gitignored). Override with `TINYCHAT_BASE_DIR` env var.

- `weights/base_data/` — FineWeb-edu parquet shards
- `weights/tokenizer/` — trained BPE tokenizer
- `weights/mlx_checkpoints/d{depth}/` — base pretrain checkpoints
- `weights/mlx_checkpoints/d{depth}_sft/` — SFT checkpoints
