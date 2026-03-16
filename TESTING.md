# Testing

## Running Tests

```bash
make test            # All tests
make test-fast       # Skip slow tests
make test-tools      # Tool calling only
make test-gpt        # GPT model only
```

Or directly with pytest:

```bash
python -m pytest tests/ -v
python -m pytest tests/ -v -m "not slow"
python -m pytest tests/test_tools.py -v
python -m pytest tests/test_gpt.py -v
```

## Test Files

| File | What it covers |
|------|---------------|
| `tests/test_gpt.py` | Forward pass, loss/gradients, memory usage, logit softcap, output shapes |
| `tests/test_tools.py` | ToolRegistry dispatch, kwarg parsing, ToolCallingTask training data |

## Tool Calling Tests (`tests/test_tools.py`)

Tests the `ToolRegistry` and `ToolCallingTask` added for general tool calling support.

**ToolRegistry dispatch:**
- Known tool with string arg → correct result
- Known tool with numeric/float args → correct result
- Unknown tool → returns `None` (engine falls back to calculator)
- Bad syntax / positional args → returns `None`
- No-arg tool → works
- Runtime exception in tool → returns `None` (never crashes engine)

**System prompt:**
- `system_prompt_block()` contains tool names and descriptions
- Empty registry returns empty string

**ToolCallingTask training data:**
- Returns valid `{"messages": [...]}` with system + user + assistant
- Assistant content is a list of `text`/`python`/`python_output` parts
- Every `python` part dispatches correctly via `_REGISTRY`
- Examples are deterministic (same index → same example)
- System prompt contains tool description block with `<|python_start|>`

## GPT Model Tests (`tests/test_gpt.py`)

Covers the core model without loading real checkpoints (uses mock classes):

- **Stage 1 — Forward pass**: output shape, logit softcap, dtype
- **Stage 2 — Loss + gradients**: loss computation, backward pass
- **Memory**: peak memory stays within expected bounds

MLX-specific tests are automatically skipped when `mlx` is not installed.

## Manual Tool Calling Test

After training a depth=4 SFT model, verify tool dispatch end-to-end:

```bash
# Word length tool
python -m scripts.chat --depth=4 --source=sft \
  -p "How many characters are in the word strawberry?"

# Arithmetic tool
python -m scripts.chat --depth=4 --source=sft \
  -p "What is 47 plus 83?"

# Reverse string tool
python -m scripts.chat --depth=4 --source=sft \
  -p "What is the word 'hello' spelled backwards?"

# No-tools mode (raw LM)
python -m scripts.chat --depth=4 --source=sft --no-tools \
  -p "What is 2 + 2?"
```

Expected output format when a tool fires:
```
I'll use the get_word_length tool.

<<get_word_length(word="strawberry")=10>>

The word 'strawberry' has 10 characters.
```

## Preview Training Data

```bash
make preview-tools
# or
python -m tasks.tool_calling
```

Prints 12 synthetic training examples across all 6 example types (word length,
reverse, arithmetic, multi-tool, palindrome, celsius).
