# ============================================================
# tinychat-mlx Makefile
# All targets use `uv run` to ensure the correct virtualenv.
# ============================================================

PYTHON = uv run python

.PHONY: test test-tools test-fast test-gpt \
        data tokenizer \
        pretrain-d4 pretrain-d12 \
        sft-d4 sft-d12 \
        train-d4 train-d12 \
        chat-d4 chat-d12 \
        eval-d4 eval-d12 \
        serve-d4 serve-d12 \
        docker-build docker-up docker-down \
        preview-tools

# ------------------------------------------------------------
# Tests
# ------------------------------------------------------------

## Run all tests
test:
	$(PYTHON) -m pytest tests/ -v

## Run only tool calling tests
test-tools:
	$(PYTHON) -m pytest tests/test_tools.py -v

## Run tests, skip slow ones
test-fast:
	$(PYTHON) -m pytest tests/ -v -m "not slow"

## Run only GPT model tests
test-gpt:
	$(PYTHON) -m pytest tests/test_gpt.py -v

# ------------------------------------------------------------
# Data & Tokenizer (shared across all depths)
# ------------------------------------------------------------

## Download 8 FineWeb-edu shards (~800MB)
data:
	$(PYTHON) -m tinychat_mlx.dataset -n 8

## Train BPE tokenizer (vocab size 32768)
tokenizer:
	$(PYTHON) -m scripts.tok_train

# ------------------------------------------------------------
# depth=4  (~5M params, ~1 min pretrain, good for pipeline testing)
# ------------------------------------------------------------

pretrain-d4:
	$(PYTHON) -m scripts.train --depth=4

sft-d4:
	$(PYTHON) -m scripts.sft --depth=4 --tool-epochs=2

## Full pipeline: data → tokenizer → pretrain → SFT (depth=4)
train-d4: data tokenizer pretrain-d4 sft-d4

chat-d4:
	$(PYTHON) -m scripts.chat --depth=4 --source=sft --interactive

eval-d4:
	$(PYTHON) -m scripts.chat_eval --depth=4

# ------------------------------------------------------------
# depth=12  (~125M params, ~1 hour pretrain, reasonable quality)
# ------------------------------------------------------------

pretrain-d12:
	$(PYTHON) -m scripts.train --depth=12

sft-d12:
	$(PYTHON) -m scripts.sft --depth=12 --tool-epochs=2

## Full pipeline: data → tokenizer → pretrain → SFT (depth=12)
train-d12: data tokenizer pretrain-d12 sft-d12

chat-d12:
	$(PYTHON) -m scripts.chat --depth=12 --source=sft --interactive

eval-d12:
	$(PYTHON) -m scripts.chat_eval --depth=12

# ------------------------------------------------------------
# HTTP inference server (native — MLX requires Apple Silicon)
# ------------------------------------------------------------

## Serve depth=4 SFT model at http://localhost:8000
serve-d4:
	$(PYTHON) -m scripts.serve --depth=4 --source=sft

## Serve depth=12 SFT model at http://localhost:8000
serve-d12:
	$(PYTHON) -m scripts.serve --depth=12 --source=sft

# ------------------------------------------------------------
# Docker (packages the server; weights mounted from ./weights)
# NOTE: MLX Metal inference does not work inside Docker containers.
#       Use native serve-d4 / serve-d12 targets for real inference.
# ------------------------------------------------------------

## Build Docker image
docker-build:
	docker build -t tinychat-mlx .

## Start server via docker-compose (mounts ./weights read-only)
docker-up:
	docker compose up

## Stop docker-compose services
docker-down:
	docker compose down

# ------------------------------------------------------------
# Misc
# ------------------------------------------------------------

## Preview 12 tool-calling training examples
preview-tools:
	$(PYTHON) -m tasks.tool_calling
