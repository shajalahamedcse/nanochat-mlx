# tinychat-mlx

Train your own chat model from scratch on Apple Silicon.

## What is this?

A self-contained MLX training framework that runs entirely on Apple Silicon. One complexity dial (`--depth`) controls everything: model size, learning rate, batch size, and training duration. The full pipeline goes from raw data download to a working chatbot with tool calling — no PyTorch required.

- Single complexity dial: `--depth` sets all hyperparameters automatically
- Full pipeline: data download, tokenizer training, pretraining, SFT, chat, evaluation
- **General tool calling**: register any Python function as a tool the model can call
- Web GUI wizard: `python -m scripts.quickstart` walks you through everything
- No PyTorch dependency (unless importing pretrained checkpoints)

## Quick Start

```bash
git clone https://github.com/your-username/tinychat-mlx.git
cd tinychat-mlx
uv sync
python -m scripts.quickstart
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser.

Or use Make:

```bash
make train-d4   # full pipeline at depth=4 (~10 min total)
make chat-d4    # interactive chat when done
```

## Import a Pretrained Model

Skip training entirely by importing a pretrained model from HuggingFace:

```bash
uv sync --extra convert   # Adds torch dependency for checkpoint conversion
python -m scripts.convert_from_hf --repo your-hf-org/your-model
```

## Full Pipeline (CLI)

```bash
# 1. Download data (8 shards, ~800MB)
python -m tinychat_mlx.dataset -n 8

# 2. Train BPE tokenizer (vocab size 32768)
python -m scripts.tok_train

# 3. Train base model
python -m scripts.train --depth=4

# 4. Supervised fine-tuning (includes tool calling data)
python -m scripts.sft --depth=4 --tool-epochs=2

# 5. Chat with your model
python -m scripts.chat --depth=4 --source=sft --interactive

# 6. Evaluate
python -m scripts.chat_eval --depth=4
```

## Tool Calling

The model learns to call registered Python functions by generating:

```
<|python_start|>tool_name(arg="value")<|python_end|>
<|output_start|>result<|output_end|>
```

Four tools are registered by default in `scripts/chat.py`: `get_word_length`,
`reverse_string`, `add`, `multiply`. Add your own:

```python
from tinychat_mlx.tools import ToolRegistry
from tinychat_mlx.engine import Engine

registry = ToolRegistry()

@registry.register(description="Look up the current weather")
def get_weather(city: str) -> str:
    return fetch_weather_api(city)

engine = Engine(model, tokenizer, tool_registry=registry)
```

Use `--no-tools` to disable tool calling entirely:

```bash
python -m scripts.chat --depth=4 --source=sft --no-tools --interactive
```

The `--tool-epochs` flag controls how much tool-calling synthetic data is mixed
into SFT training (default: 2 epochs of 50k examples each):

```bash
python -m scripts.sft --depth=12 --tool-epochs=4
```

## The Depth Dial

The `--depth` parameter is the single complexity dial. All other hyperparameters
(width, heads, batch size, learning rate, training tokens) are auto-computed from
depth via scaling laws.

| Depth | Params | Time (M3 Pro) | Use case |
|-------|--------|---------------|----------|
| 4     | ~5M    | ~1 min        | Quick test, debugging |
| 12    | ~125M  | ~1 hour       | Reasonable quality |
| 20    | ~350M  | ~8 hours      | Good quality |
| 26    | ~600M  | ~24 hours     | GPT-2 reproduction |

The "miniseries principle" requires any architectural change to work across all depths.

## Hardware Requirements

Apple Silicon is required (M1, M2, M3, M4 — any variant).

Recommended RAM by depth:

- **8 GB** — depth 4 (quick tests and debugging)
- **16 GB** — depth 12 (reasonable quality training)
- **32 GB+** — depth 20 and above

## Project Structure

```
tinychat_mlx/          Core MLX modules
  gpt.py               GPT transformer model
  optim.py             Muon+AdamW optimizer
  engine.py            Inference with KV cache and tool dispatch
  train.py             Training loop
  sft.py               SFT pipeline
  tools.py             ToolRegistry for general tool calling
  eval.py              BPB evaluation
  dataloader.py        BOS-aligned best-fit packing
  sft_dataloader.py    SFT conversation packing
  dataset.py           Data download and iteration
  tokenizer.py         BPE tokenizer
  common.py            Memory management, utilities
scripts/               Entry points
  quickstart.py        Web GUI wizard
  train.py             Training CLI
  sft.py               SFT CLI (--tool-epochs flag)
  chat.py              Chat CLI (--no-tools flag)
  chat_eval.py         Evaluation CLI
  tok_train.py         Tokenizer training
  convert_from_hf.py   HuggingFace checkpoint import
tasks/                 Training tasks
  tool_calling.py      Synthetic tool-calling training data
  spellingbee.py       Spelling and counting tasks
  gsm8k.py             Grade school math
  mmlu.py              MMLU benchmark
tests/                 Test suite
runs/                  Shell scripts
```

## HTTP Server

Serve the model over HTTP with an OpenAI-compatible API:

```bash
# Native (required for real MLX inference)
make serve-d4               # http://localhost:8000
python -m scripts.serve --depth=12 --source=sft --port=8080

# Endpoints
GET  /health                  model info + loaded tools
GET  /v1/models               list model
POST /v1/chat/completions     chat (supports stream=true)
GET  /tools                   list registered tools
```

Example request:
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "How many letters in strawberry?"}], "stream": false}'
```

Streaming:
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "What is 47 plus 83?"}], "stream": true}'
```

**Docker**: packages the server for distribution. Metal GPU is not accessible
from inside Docker containers, so inference won't work there. Mount `weights/`
from the host and run natively instead.

```bash
make docker-build
make docker-up      # mounts ./weights read-only
make docker-down
```

## Weights & Data

All artifacts (data shards, tokenizer, checkpoints) are stored in a local `weights/`
folder at the project root (gitignored). Override the location:

```bash
TINYCHAT_BASE_DIR=/path/to/storage python -m scripts.train --depth=12
```

The layout inside `weights/`:
```
weights/
  base_data/                    FineWeb-edu parquet shards
  tokenizer/                    trained BPE tokenizer
  mlx_checkpoints/d4/           base pretrain checkpoint
  mlx_checkpoints/d4_sft/       SFT checkpoint
```

## Tests

See [TESTING.md](TESTING.md) for the full testing guide.

```bash
make test            # All tests
make test-tools      # Tool calling only
make test-fast       # Skip slow tests
```

