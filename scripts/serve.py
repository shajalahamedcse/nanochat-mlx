"""
HTTP inference server for tinychat-mlx.

OpenAI-compatible /v1/chat/completions endpoint with streaming support
and tool calling. Model is loaded once at startup.

Usage:
    python -m scripts.serve --depth=4 --source=sft
    python -m scripts.serve --depth=12 --source=sft --port=8080 --host=0.0.0.0

Endpoints:
    GET  /health                  model info + readiness
    GET  /v1/models               list loaded model (OpenAI-compatible)
    POST /v1/chat/completions     chat with optional streaming
    GET  /tools                   list registered tools

Note: MLX requires Apple Silicon (Metal GPU). Run the server natively on
macOS — do not expect GPU inference inside a Docker container. The Dockerfile
packages the server for distribution; mount weights/ from the host and run
with --platform linux/arm64 or natively.
"""

import argparse
import asyncio
import json
import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncIterator

import mlx.core as mx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import uvicorn

from tinychat_mlx.common import set_memory_limit, get_base_dir
from tinychat_mlx.engine import Engine
from tinychat_mlx.tools import ToolRegistry


# ---------------------------------------------------------------------------
# Arg parsing (runs at import time so uvicorn picks them up)
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="tinychat-mlx HTTP inference server")
parser.add_argument("--depth", type=int, default=4, help="Model depth to load")
parser.add_argument("--step", type=int, default=None, help="Checkpoint step (default: latest)")
parser.add_argument("--source", type=str, default="sft", choices=["base", "sft"],
                    help="Checkpoint source: base or sft (default: sft)")
parser.add_argument("--host", type=str, default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
parser.add_argument("--port", type=int, default=8000, help="Port (default: 8000)")
parser.add_argument("--max-tokens", type=int, default=512, help="Default max tokens")
parser.add_argument("--temperature", type=float, default=0.8, help="Default temperature")
parser.add_argument("--top-k", type=int, default=50, help="Default top-k")
parser.add_argument("--memory-limit-gb", type=float, default=16.0, help="MLX memory limit in GB")
parser.add_argument("--no-tools", action="store_true", help="Disable tool registry")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# App + globals
# ---------------------------------------------------------------------------

_engine: Engine | None = None
_tokenizer = None
_model_id: str = f"tinychat-d{args.depth}-{args.source}"


# ---------------------------------------------------------------------------
# Model loading — lifespan handler (FastAPI modern style)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _engine, _tokenizer

    set_memory_limit(args.memory_limit_gb)

    from tinychat_mlx.tokenizer import get_tokenizer
    from scripts.chat import load_model as _load_model

    _tokenizer = get_tokenizer()
    model = _load_model(depth=args.depth, step=args.step, source=args.source)

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

    _engine = Engine(model, _tokenizer, tool_registry=tool_registry)
    print(f"Model ready: {_model_id}")
    yield  # server runs here


app = FastAPI(
    title="tinychat-mlx",
    description="tinychat-mlx inference server",
    version="0.1.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = Field(default="tinychat")
    messages: list[ChatMessage]
    stream: bool = False
    max_tokens: int | None = None
    temperature: float | None = None
    top_k: int | None = None


# ---------------------------------------------------------------------------
# Token generation helpers
# ---------------------------------------------------------------------------

def _build_prompt(messages: list[ChatMessage]) -> list[int]:
    """Convert chat messages to token IDs primed for assistant completion."""
    bos_id = _tokenizer.get_bos_token_id()
    tool_registry = _engine.tool_registry

    # Build system block from tools if available
    system_block = ""
    if tool_registry and tool_registry.list_tools():
        system_block = tool_registry.system_prompt_block()

    # Merge system message + tool block into first user turn
    parts = []
    for msg in messages:
        if msg.role == "system":
            system_block = (msg.content + "\n\n" + system_block).strip() if system_block else msg.content
        else:
            parts.append(msg)

    # Render conversation tokens
    conversation_messages = []
    if system_block:
        conversation_messages.append({"role": "system", "content": system_block})
    for msg in parts:
        conversation_messages.append({"role": msg.role, "content": msg.content})

    # Add empty assistant turn to prime generation
    conversation_messages.append({"role": "assistant", "content": ""})

    # Use render_conversation to get token ids; strip the trailing assistant_end
    ids, _ = _tokenizer.render_conversation({"messages": conversation_messages})
    # Remove the trailing <|assistant_end|> that render_conversation adds
    # so the model generates the assistant response from scratch
    assistant_end = _tokenizer.encode_special("<|assistant_end|>")
    if ids and ids[-1] == assistant_end:
        ids = ids[:-1]
    return ids


def _decode_tokens_clean(token_ids: list[int]) -> str:
    """
    Decode token IDs to text, suppressing:
    - Special tokens (<|...|>)
    - Tool call expressions inside <|python_start|>...<|python_end|>
    Tool outputs (<|output_start|>...<|output_end|>) are rendered as
    [tool: result] so the caller can see what the tool returned.
    """
    python_start = _tokenizer.encode_special("<|python_start|>")
    python_end = _tokenizer.encode_special("<|python_end|>")
    output_start = _tokenizer.encode_special("<|output_start|>")
    output_end = _tokenizer.encode_special("<|output_end|>")

    in_python = False
    in_output = False
    output_buf: list[int] = []
    parts: list[str] = []

    for t in token_ids:
        if t == python_start:
            in_python = True
        elif t == python_end:
            in_python = False
        elif t == output_start:
            in_output = True
            output_buf = []
        elif t == output_end:
            in_output = False
            result = _tokenizer.decode(output_buf).strip()
            parts.append(f"[tool: {result}]")
        elif in_python or in_output:
            if in_output:
                output_buf.append(t)
            # skip python expression tokens
        else:
            text = _tokenizer.decode([t])
            if not text.startswith("<|"):
                parts.append(text)

    return "".join(parts)


async def _stream_tokens(tokens: list[int], max_tokens: int, temperature: float, top_k: int) -> AsyncIterator[str]:
    """Async generator that yields SSE chunks in OpenAI format."""
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    created = int(time.time())

    python_start = _tokenizer.encode_special("<|python_start|>")
    python_end = _tokenizer.encode_special("<|python_end|>")
    output_start = _tokenizer.encode_special("<|output_start|>")
    output_end = _tokenizer.encode_special("<|output_end|>")

    in_python = False
    in_output = False
    output_buf: list[int] = []

    def _make_chunk(delta_content: str, finish_reason=None) -> str:
        chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": _model_id,
            "choices": [{
                "index": 0,
                "delta": {"content": delta_content} if delta_content else {},
                "finish_reason": finish_reason,
            }],
        }
        return f"data: {json.dumps(chunk)}\n\n"

    # Yield role header
    yield f"data: {json.dumps({'id': completion_id, 'object': 'chat.completion.chunk', 'created': created, 'model': _model_id, 'choices': [{'index': 0, 'delta': {'role': 'assistant'}, 'finish_reason': None}]})}\n\n"

    for token_column, token_masks in _engine.generate(
        tokens,
        num_samples=1,
        max_tokens=max_tokens,
        temperature=temperature,
        top_k=top_k,
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
            result = _tokenizer.decode(output_buf).strip()
            yield _make_chunk(f"[tool: {result}]")
        elif in_python:
            pass  # suppress tool call expression
        elif in_output:
            output_buf.append(token)
        else:
            text = _tokenizer.decode([token])
            if text and not text.startswith("<|"):
                yield _make_chunk(text)
        await asyncio.sleep(0)

    yield _make_chunk("", finish_reason="stop")
    yield "data: [DONE]\n\n"


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": _model_id,
        "depth": args.depth,
        "source": args.source,
        "tools": [t.name for t in _engine.tool_registry.list_tools()] if _engine and _engine.tool_registry else [],
    }


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [{
            "id": _model_id,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "tinychat-mlx",
        }],
    }


@app.get("/tools")
def list_tools():
    if not _engine or not _engine.tool_registry:
        return {"tools": []}
    return {
        "tools": [
            {"name": t.name, "description": t.description, "signature": t.signature_str()}
            for t in _engine.tool_registry.list_tools()
        ]
    }


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest):
    if _engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    max_tokens = req.max_tokens or args.max_tokens
    temperature = req.temperature if req.temperature is not None else args.temperature
    top_k = req.top_k or args.top_k

    tokens = _build_prompt(req.messages)

    if req.stream:
        return StreamingResponse(
            _stream_tokens(tokens, max_tokens, temperature, top_k),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # Non-streaming: collect full response
    output_tokens = []
    for token_column, _ in _engine.generate(
        tokens, num_samples=1, max_tokens=max_tokens,
        temperature=temperature, top_k=top_k,
    ):
        output_tokens.append(token_column[0])

    content = _decode_tokens_clean(output_tokens)

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": _model_id,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": len(tokens),
            "completion_tokens": len(output_tokens),
            "total_tokens": len(tokens) + len(output_tokens),
        },
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(app, host=args.host, port=args.port)
