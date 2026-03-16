"""
Tool registry for tinychat-mlx.

Lets users register Python functions as named tools. The model calls them by
generating:
    <|python_start|>tool_name(arg="value")<|python_end|>

The engine dispatches to the right tool and injects the result as:
    <|output_start|>result<|output_end|>

Dispatch uses ast.literal_eval per argument value — never eval() on the full
expression — so arbitrary code cannot be injected via tool arguments.
"""

import ast
import inspect
import re
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class ToolSpec:
    name: str
    fn: Callable
    description: str
    param_descriptions: dict[str, str] = field(default_factory=dict)

    def signature_str(self) -> str:
        sig = inspect.signature(self.fn)
        params = []
        for pname, param in sig.parameters.items():
            ann = param.annotation
            type_str = ann.__name__ if ann is not inspect.Parameter.empty else "any"
            params.append(f"{pname}: {type_str}")
        return f"{self.name}({', '.join(params)})"

    def prompt_line(self) -> str:
        line = f"- {self.signature_str()}: {self.description}"
        if self.param_descriptions:
            extras = ", ".join(f"{k}: {v}" for k, v in self.param_descriptions.items())
            line += f" ({extras})"
        return line


# Matches the outermost function call: name(args...)
_CALL_RE = re.compile(r"^(\w+)\((.*)\)$", re.DOTALL)


class ToolRegistry:
    """
    Registry of Python functions callable by the model.

    Usage (decorator form):
        registry = ToolRegistry()

        @registry.register(description="Add two numbers")
        def add(a: float, b: float) -> str:
            return str(a + b)

    Usage (imperative form):
        registry.register_fn(my_fn, description="Does something")
    """

    def __init__(self):
        self._tools: dict[str, ToolSpec] = {}

    def register(
        self,
        description: str,
        param_descriptions: dict[str, str] | None = None,
    ) -> Callable:
        """Decorator form of registration."""
        def decorator(fn: Callable) -> Callable:
            self.register_fn(fn, description=description,
                             param_descriptions=param_descriptions or {})
            return fn
        return decorator

    def register_fn(
        self,
        fn: Callable,
        description: str,
        param_descriptions: dict[str, str] | None = None,
    ) -> None:
        """Imperative form of registration."""
        self._tools[fn.__name__] = ToolSpec(
            name=fn.__name__,
            fn=fn,
            description=description,
            param_descriptions=param_descriptions or {},
        )

    def list_tools(self) -> list[ToolSpec]:
        return list(self._tools.values())

    def system_prompt_block(self) -> str:
        """
        Returns text describing available tools, suitable for prepending to
        the system/user message so the model knows what it can call.
        """
        if not self._tools:
            return ""
        lines = [
            "You have access to the following tools. Call them by writing",
            '<|python_start|>tool_name(arg="value")<|python_end|>',
            "and the result will be provided as <|output_start|>result<|output_end|>.\n",
            "Available tools:",
        ]
        for spec in self._tools.values():
            lines.append(spec.prompt_line())
        return "\n".join(lines)

    def dispatch(self, expr: str) -> str | None:
        """
        Parse expr as a Python-style function call and dispatch to the
        registered tool. Returns the string result, or None on any failure
        (unknown tool, parse error, runtime exception).

        Falling through to None lets the engine try use_calculator() as a
        fallback for raw arithmetic expressions like "2 + 2".
        """
        expr = expr.strip()
        m = _CALL_RE.match(expr)
        if not m:
            return None
        tool_name, args_str = m.group(1), m.group(2).strip()
        spec = self._tools.get(tool_name)
        if spec is None:
            return None
        kwargs = _parse_kwargs(args_str)
        if kwargs is None:
            return None
        try:
            result = spec.fn(**kwargs)
            return str(result) if result is not None else None
        except Exception:
            return None


def _parse_kwargs(args_str: str) -> dict[str, Any] | None:
    """
    Parse 'key="val", key2=42' into {'key': 'val', 'key2': 42}.
    Uses ast.literal_eval per value — never evals the full expression.
    Returns None on any parse failure.
    """
    if not args_str.strip():
        return {}
    tokens = _split_args(args_str)
    if tokens is None:
        return None
    result: dict[str, Any] = {}
    for token in tokens:
        token = token.strip()
        if "=" not in token:
            return None  # positional args not supported
        key, _, val_str = token.partition("=")
        key = key.strip()
        if not key.isidentifier():
            return None
        try:
            result[key] = ast.literal_eval(val_str.strip())
        except (ValueError, SyntaxError):
            return None
    return result


def _split_args(s: str) -> list[str] | None:
    """Split comma-separated arg string respecting string literals and brackets."""
    parts: list[str] = []
    depth = 0
    in_str = False
    str_char: str | None = None
    current: list[str] = []
    i = 0
    while i < len(s):
        ch = s[i]
        if in_str:
            current.append(ch)
            if ch == "\\" and i + 1 < len(s):
                # escaped character inside string — skip next char
                i += 1
                current.append(s[i])
            elif ch == str_char:
                in_str = False
        elif ch in ('"', "'"):
            in_str = True
            str_char = ch
            current.append(ch)
        elif ch in ("(", "[", "{"):
            depth += 1
            current.append(ch)
        elif ch in (")", "]", "}"):
            depth -= 1
            current.append(ch)
        elif ch == "," and depth == 0:
            parts.append("".join(current))
            current = []
        else:
            current.append(ch)
        i += 1
    if in_str:
        return None  # unterminated string
    if current or parts:
        parts.append("".join(current))
    return parts
