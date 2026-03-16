"""
Tests for the ToolRegistry and ToolCallingTask.

Run with:
    make test-tools
    python -m pytest tests/test_tools.py -v
"""

import pytest
from tinychat_mlx.tools import ToolRegistry, _parse_kwargs, _split_args


# ---------------------------------------------------------------------------
# ToolRegistry tests
# ---------------------------------------------------------------------------

def _make_registry():
    registry = ToolRegistry()

    @registry.register(description="Returns the length of a word")
    def get_word_length(word: str) -> str:
        return str(len(word))

    @registry.register(description="Adds two numbers")
    def add(a: float, b: float) -> str:
        return str(a + b)

    return registry


def test_dispatch_known_tool_string_arg():
    r = _make_registry()
    assert r.dispatch('get_word_length(word="hello")') == "5"


def test_dispatch_known_tool_numeric_args():
    r = _make_registry()
    # ast.literal_eval parses "3" as int, so 3+4=7 (int), not 7.0
    assert r.dispatch("add(a=3, b=4)") == "7"


def test_dispatch_known_tool_float_args():
    r = _make_registry()
    result = r.dispatch("add(a=1.5, b=2.5)")
    assert result == "4.0"


def test_dispatch_unknown_tool_returns_none():
    r = _make_registry()
    assert r.dispatch('search(query="test")') is None


def test_dispatch_bad_syntax_returns_none():
    r = _make_registry()
    assert r.dispatch("not a function call") is None
    assert r.dispatch("") is None
    assert r.dispatch("get_word_length(hello)") is None  # positional arg


def test_dispatch_no_args():
    r = ToolRegistry()

    @r.register(description="Returns a greeting")
    def greet() -> str:
        return "hello"

    assert r.dispatch("greet()") == "hello"


def test_dispatch_runtime_error_returns_none():
    r = ToolRegistry()

    @r.register(description="Always raises")
    def boom(x: str) -> str:
        raise ValueError("boom")

    assert r.dispatch('boom(x="test")') is None


def test_system_prompt_block_contains_tool_info():
    r = _make_registry()
    block = r.system_prompt_block()
    assert "get_word_length" in block
    assert "add" in block
    assert "Returns the length of a word" in block
    assert "<|python_start|>" in block


def test_system_prompt_block_empty_registry():
    r = ToolRegistry()
    assert r.system_prompt_block() == ""


def test_list_tools():
    r = _make_registry()
    names = [t.name for t in r.list_tools()]
    assert "get_word_length" in names
    assert "add" in names


# ---------------------------------------------------------------------------
# _parse_kwargs tests
# ---------------------------------------------------------------------------

def test_parse_kwargs_string():
    result = _parse_kwargs('word="hello"')
    assert result == {"word": "hello"}


def test_parse_kwargs_int():
    result = _parse_kwargs("n=42")
    assert result == {"n": 42}


def test_parse_kwargs_float():
    result = _parse_kwargs("celsius=23.5")
    assert result == {"celsius": 23.5}


def test_parse_kwargs_negative_float():
    result = _parse_kwargs("celsius=-10.0")
    assert result == {"celsius": -10.0}


def test_parse_kwargs_multiple():
    result = _parse_kwargs('a=1, b="two"')
    assert result == {"a": 1, "b": "two"}


def test_parse_kwargs_empty():
    assert _parse_kwargs("") == {}
    assert _parse_kwargs("   ") == {}


def test_parse_kwargs_positional_returns_none():
    assert _parse_kwargs('"hello"') is None


# ---------------------------------------------------------------------------
# ToolCallingTask tests
# ---------------------------------------------------------------------------

def test_tool_calling_task_returns_valid_conversation():
    from tasks.tool_calling import ToolCallingTask
    task = ToolCallingTask(size=100, split="train")
    ex = task.get_example(0)
    assert "messages" in ex
    messages = ex["messages"]
    # system + user + assistant
    assert len(messages) == 3
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert messages[2]["role"] == "assistant"


def test_tool_calling_task_assistant_parts():
    from tasks.tool_calling import ToolCallingTask
    task = ToolCallingTask(size=100, split="train")
    for i in range(6):  # covers all 6 example types
        ex = task.get_example(i)
        content = ex["messages"][2]["content"]
        assert isinstance(content, list), f"example {i}: content should be a list"
        types = {part["type"] for part in content}
        assert "python" in types, f"example {i}: missing 'python' part"
        assert "python_output" in types, f"example {i}: missing 'python_output' part"
        assert "text" in types, f"example {i}: missing 'text' part"


def test_tool_calling_task_python_call_is_valid():
    """Every python part should parse as a valid tool dispatch."""
    from tasks.tool_calling import ToolCallingTask, _REGISTRY
    task = ToolCallingTask(size=60, split="train")
    for i in range(6):
        ex = task.get_example(i)
        for part in ex["messages"][2]["content"]:
            if part["type"] == "python":
                result = _REGISTRY.dispatch(part["text"])
                assert result is not None, (
                    f"example {i}: dispatch failed for: {part['text']!r}"
                )


def test_tool_calling_task_deterministic():
    from tasks.tool_calling import ToolCallingTask
    task = ToolCallingTask(size=100, split="train")
    ex1 = task.get_example(7)
    ex2 = task.get_example(7)
    assert ex1 == ex2


def test_tool_calling_task_system_prompt_has_tool_description():
    from tasks.tool_calling import ToolCallingTask
    task = ToolCallingTask(size=100, split="train")
    ex = task.get_example(0)
    system_content = ex["messages"][0]["content"]
    assert "<|python_start|>" in system_content
    assert "Available tools:" in system_content


def test_tool_calling_task_num_examples():
    from tasks.tool_calling import ToolCallingTask
    task = ToolCallingTask(size=50, split="train")
    assert task.num_examples() == 50
