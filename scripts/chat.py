"""
Chat CLI using MLX inference engine.

Usage:
    python -m scripts.chat -p "Why is the sky blue?"
    python -m scripts.chat --interactive
"""

import os
import argparse

import mlx.core as mx

from tinychat_mlx.gpt import GPT, GPTConfig
from tinychat_mlx.train import _load_weights_into_model
from tinychat_mlx.engine import Engine
from tinychat_mlx.common import print0, get_base_dir, set_memory_limit
from tinychat_mlx.tools import ToolRegistry


def load_model(depth=12, step=None, source="base"):
    """Load a trained MLX model from checkpoint.

    Args:
        depth: model depth
        step: checkpoint step (None = latest)
        source: "base" for pretrained, "sft" for fine-tuned
    """
    base_dir = get_base_dir()
    if source == "sft":
        ckpt_dir = os.path.join(base_dir, "mlx_checkpoints", f"d{depth}_sft")
    else:
        ckpt_dir = os.path.join(base_dir, "mlx_checkpoints", f"d{depth}")

    if not os.path.exists(ckpt_dir):
        raise FileNotFoundError(f"No checkpoint directory found at {ckpt_dir}")

    # Find latest checkpoint if step not specified
    if step is None:
        safetensor_files = sorted([
            f for f in os.listdir(ckpt_dir)
            if f.endswith(".safetensors") and not f.endswith("_optim.safetensors")
        ])
        if not safetensor_files:
            raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}")
        latest = safetensor_files[-1]
        weights_path = os.path.join(ckpt_dir, latest)
        meta_path = weights_path.replace(".safetensors", "_meta.json")
    else:
        weights_path = os.path.join(ckpt_dir, f"step_{step:06d}.safetensors")
        meta_path = os.path.join(ckpt_dir, f"step_{step:06d}_meta.json")

    # Load metadata
    import json
    with open(meta_path) as f:
        meta = json.load(f)

    print0(f"Loading checkpoint: {weights_path}")
    print0(f"  depth={meta['depth']}, n_embd={meta['n_embd']}, step={meta['step']}")

    # Build model
    config = GPTConfig(
        sequence_len=meta["sequence_len"],
        vocab_size=meta["vocab_size"],
        n_layer=meta["depth"],
        n_head=meta["n_head"],
        n_kv_head=meta["n_kv_head"],
        n_embd=meta["n_embd"],
        window_pattern=meta.get("window_pattern", "L"),
    )
    model = GPT(config)
    _load_weights_into_model(model, weights_path)
    mx.eval(model.parameters())
    return model


def main():
    parser = argparse.ArgumentParser(description="Chat with MLX model")
    parser.add_argument("-p", "--prompt", type=str, default=None, help="single prompt")
    parser.add_argument("--interactive", action="store_true", help="interactive chat mode")
    parser.add_argument("--depth", type=int, default=12, help="model depth to load")
    parser.add_argument("--step", type=int, default=None, help="checkpoint step (default: latest)")
    parser.add_argument("--source", type=str, default="base", choices=["base", "sft"],
                        help="checkpoint source: base or sft (default: base)")
    parser.add_argument("--max-tokens", type=int, default=256, help="max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="sampling temperature")
    parser.add_argument("--top-k", type=int, default=50, help="top-k sampling")
    parser.add_argument("--memory-limit-gb", type=float, default=16.0, help="MLX memory limit")
    parser.add_argument("--no-tools", action="store_true", help="disable tool registry (raw LM mode)")
    args = parser.parse_args()

    set_memory_limit(args.memory_limit_gb)

    # Load model and tokenizer
    from tinychat_mlx.tokenizer import get_tokenizer
    tokenizer = get_tokenizer()
    model = load_model(depth=args.depth, step=args.step, source=args.source)
    bos_id = tokenizer.get_bos_token_id()

    tool_registry = None
    if not args.no_tools:
        tool_registry = ToolRegistry()

        @tool_registry.register(description="Returns the number of characters in a string")
        def get_word_length(word: str) -> str:
            return str(len(word))

        @tool_registry.register(description="Reverses a string")
        def reverse_string(text: str) -> str:
            return text[::-1]

        @tool_registry.register(description="Adds two numbers")
        def add(a: float, b: float) -> str:
            return str(a + b)

        @tool_registry.register(description="Multiplies two numbers")
        def multiply(a: float, b: float) -> str:
            return str(a * b)

    engine = Engine(model, tokenizer, tool_registry=tool_registry)

    python_start = tokenizer.encode_special("<|python_start|>")
    python_end   = tokenizer.encode_special("<|python_end|>")
    output_start = tokenizer.encode_special("<|output_start|>")
    output_end   = tokenizer.encode_special("<|output_end|>")

    def generate_response(prompt):
        if tool_registry is not None and tool_registry.list_tools():
            full_prompt = tool_registry.system_prompt_block() + "\n\n" + prompt
        else:
            full_prompt = prompt
        tokens = tokenizer.encode(full_prompt, prepend=bos_id)
        in_python = False
        in_output = False
        output_buf = []
        for token_column, token_masks in engine.generate(
            tokens, num_samples=1,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        ):
            token = token_column[0]
            if token == python_start:
                in_python = True
            elif token == python_end:
                in_python = False
            elif token == output_start:
                in_output = True
                output_buf = []
            elif token == output_end:
                in_output = False
                result = tokenizer.decode(output_buf).strip()
                print(f"[tool: {result}]", end="", flush=True)
            elif in_python:
                pass  # suppress raw tool call expression
            elif in_output:
                output_buf.append(token)
            else:
                chunk = tokenizer.decode([token])
                if chunk and not chunk.startswith("<|"):
                    print(chunk, end="", flush=True)
        print()

    if args.prompt:
        generate_response(args.prompt)
    elif args.interactive:
        print0("Interactive chat mode. Type 'quit' to exit.\n")
        while True:
            try:
                prompt = input("You: ").strip()
                if prompt.lower() in ("quit", "exit", "q"):
                    break
                if not prompt:
                    continue
                print("Assistant: ", end="")
                generate_response(prompt)
                print()
            except (KeyboardInterrupt, EOFError):
                print()
                break
    else:
        # Default prompt
        generate_response("The capital of France is")


if __name__ == "__main__":
    main()
