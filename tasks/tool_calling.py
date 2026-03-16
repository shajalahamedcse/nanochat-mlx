"""
Synthetic tool-calling training data for tinychat-mlx.

Generates conversations demonstrating the tool calling format:
    <|python_start|>tool_name(arg="value")<|python_end|>
    <|output_start|>result<|output_end|>

Follows the exact structure of tasks/spellingbee.py:
- Deterministic via random.Random(seed_offset + index)
- Reuses the same word list URL
- Returns {"messages": [...]} with assistant content as a list of parts

To preview examples:
    python -m tasks.tool_calling
"""

import random
from tasks.common import Task
from tinychat_mlx.common import download_file_with_lock
from tinychat_mlx.tools import ToolRegistry

WORD_LIST_URL = "https://raw.githubusercontent.com/dwyl/english-words/refs/heads/master/words_alpha.txt"
TEST_RANDOM_SEED_OFFSET = 10_000_000


# ---------------------------------------------------------------------------
# Shared registry used for both training data generation and dispatch tests.
# Maps to pure Python functions whose outputs we know at data-gen time.
# ---------------------------------------------------------------------------

_REGISTRY = ToolRegistry()


@_REGISTRY.register(
    description="Returns the number of characters in a word",
    param_descriptions={"word": "the word to measure"},
)
def get_word_length(word: str) -> str:
    return str(len(word))


@_REGISTRY.register(
    description="Reverses a string",
    param_descriptions={"text": "the string to reverse"},
)
def reverse_string(text: str) -> str:
    return text[::-1]


@_REGISTRY.register(
    description="Returns True if the string is a palindrome, False otherwise",
    param_descriptions={"text": "the string to check"},
)
def is_palindrome(text: str) -> str:
    t = text.lower()
    return str(t == t[::-1])


@_REGISTRY.register(
    description="Adds two numbers",
    param_descriptions={"a": "first number", "b": "second number"},
)
def add(a: float, b: float) -> str:
    return str(a + b)


@_REGISTRY.register(
    description="Multiplies two numbers",
    param_descriptions={"a": "first number", "b": "second number"},
)
def multiply(a: float, b: float) -> str:
    return str(a * b)


@_REGISTRY.register(
    description="Counts vowels in a string",
    param_descriptions={"text": "the string to count vowels in"},
)
def count_vowels(text: str) -> str:
    return str(sum(1 for c in text.lower() if c in "aeiou"))


@_REGISTRY.register(
    description="Converts a Celsius temperature to Fahrenheit",
    param_descriptions={"celsius": "temperature in Celsius"},
)
def celsius_to_fahrenheit(celsius: float) -> str:
    return str(round(celsius * 9 / 5 + 32, 2))


# ---------------------------------------------------------------------------
# Per-tool system prompt fragments (only the relevant tools are shown per ex)
# ---------------------------------------------------------------------------

def _system_prompt(*tool_names: str) -> str:
    lines = [
        "You have access to the following tools. Call them by writing",
        '<|python_start|>tool_name(arg="value")<|python_end|>',
        "and the result will be provided as <|output_start|>result<|output_end|>.\n",
        "Available tools:",
    ]
    for spec in _REGISTRY.list_tools():
        if spec.name in tool_names:
            lines.append(spec.prompt_line())
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Example generators
# ---------------------------------------------------------------------------

def _ex_word_length(rng: random.Random, words: list[str]) -> dict:
    word = rng.choice(words)
    length = len(word)
    user_msg = rng.choice([
        f"How many characters are in the word '{word}'?",
        f"What is the length of '{word}'?",
        f"How long is the word {word}?",
        f"Count the characters in '{word}'.",
    ])
    call = f'get_word_length(word="{word}")'
    parts = [
        {"type": "text", "text": f"I'll use the get_word_length tool.\n\n"},
        {"type": "python", "text": call},
        {"type": "python_output", "text": str(length)},
        {"type": "text", "text": f"\n\nThe word '{word}' has {length} characters."},
    ]
    return {"messages": [
        {"role": "system", "content": _system_prompt("get_word_length")},
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": parts},
    ]}


def _ex_reverse(rng: random.Random, words: list[str]) -> dict:
    text = rng.choice(words)
    reversed_text = text[::-1]
    user_msg = rng.choice([
        f"What is '{text}' backwards?",
        f"Reverse the string '{text}'.",
        f"Spell '{text}' in reverse.",
    ])
    call = f'reverse_string(text="{text}")'
    parts = [
        {"type": "text", "text": f"Let me reverse '{text}'.\n\n"},
        {"type": "python", "text": call},
        {"type": "python_output", "text": reversed_text},
        {"type": "text", "text": f"\n\n'{text}' reversed is '{reversed_text}'."},
    ]
    return {"messages": [
        {"role": "system", "content": _system_prompt("reverse_string")},
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": parts},
    ]}


def _ex_arithmetic(rng: random.Random) -> dict:
    op = rng.choice(["add", "multiply"])
    a = rng.randint(1, 999)
    b = rng.randint(1, 999)
    if op == "add":
        result_val = str(float(a + b))
        user_msg = rng.choice([
            f"What is {a} plus {b}?",
            f"Add {a} and {b}.",
            f"Calculate {a} + {b}.",
        ])
        call = f"add(a={a}, b={b})"
    else:
        result_val = str(float(a * b))
        user_msg = rng.choice([
            f"What is {a} times {b}?",
            f"Multiply {a} by {b}.",
            f"Calculate {a} × {b}.",
        ])
        call = f"multiply(a={a}, b={b})"
    parts = [
        {"type": "text", "text": f"I'll use the {op} tool.\n\n"},
        {"type": "python", "text": call},
        {"type": "python_output", "text": result_val},
        {"type": "text", "text": f"\n\nThe result is {result_val}."},
    ]
    return {"messages": [
        {"role": "system", "content": _system_prompt("add", "multiply")},
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": parts},
    ]}


def _ex_multi_tool(rng: random.Random, words: list[str]) -> dict:
    word = rng.choice(words)
    length = len(word)
    vowels = sum(1 for c in word.lower() if c in "aeiou")
    user_msg = f"For the word '{word}', tell me its length and how many vowels it has."
    parts = [
        {"type": "text", "text": f"I'll check both properties of '{word}'.\n\nFirst, the length:\n\n"},
        {"type": "python", "text": f'get_word_length(word="{word}")'},
        {"type": "python_output", "text": str(length)},
        {"type": "text", "text": "\n\nNow the vowel count:\n\n"},
        {"type": "python", "text": f'count_vowels(text="{word}")'},
        {"type": "python_output", "text": str(vowels)},
        {"type": "text", "text": f"\n\nThe word '{word}' has {length} characters and {vowels} vowel(s)."},
    ]
    return {"messages": [
        {"role": "system", "content": _system_prompt("get_word_length", "count_vowels")},
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": parts},
    ]}


def _ex_palindrome(rng: random.Random, words: list[str]) -> dict:
    word = rng.choice(words)
    result = is_palindrome(word)
    user_msg = rng.choice([
        f"Is '{word}' a palindrome?",
        f"Does '{word}' read the same forwards and backwards?",
        f"Check if {word} is a palindrome.",
    ])
    call = f'is_palindrome(text="{word}")'
    verdict = "is" if result == "True" else "is not"
    parts = [
        {"type": "text", "text": f"Let me check if '{word}' is a palindrome.\n\n"},
        {"type": "python", "text": call},
        {"type": "python_output", "text": result},
        {"type": "text", "text": f"\n\n'{word}' {verdict} a palindrome."},
    ]
    return {"messages": [
        {"role": "system", "content": _system_prompt("is_palindrome")},
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": parts},
    ]}


def _ex_celsius(rng: random.Random) -> dict:
    celsius = round(rng.uniform(-40, 100), 1)
    fahrenheit_val = round(celsius * 9 / 5 + 32, 2)
    user_msg = rng.choice([
        f"Convert {celsius}°C to Fahrenheit.",
        f"What is {celsius} degrees Celsius in Fahrenheit?",
        f"What's {celsius}°C in °F?",
    ])
    call = f"celsius_to_fahrenheit(celsius={celsius})"
    parts = [
        {"type": "text", "text": f"I'll convert {celsius}°C to Fahrenheit.\n\n"},
        {"type": "python", "text": call},
        {"type": "python_output", "text": str(fahrenheit_val)},
        {"type": "text", "text": f"\n\n{celsius}°C is {fahrenheit_val}°F."},
    ]
    return {"messages": [
        {"role": "system", "content": _system_prompt("celsius_to_fahrenheit")},
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": parts},
    ]}


# ---------------------------------------------------------------------------
# Task class
# ---------------------------------------------------------------------------

_GENERATORS = [
    _ex_word_length,   # 0
    _ex_reverse,       # 1
    _ex_arithmetic,    # 2
    _ex_multi_tool,    # 3
    _ex_palindrome,    # 4
    _ex_celsius,       # 5
]
_N_TYPES = len(_GENERATORS)


class ToolCallingTask(Task):
    """
    Synthetic tool-calling training data.

    Each example is generated deterministically from its index using a seeded
    RNG, matching the pattern from tasks/spellingbee.py.
    """

    def __init__(self, size: int = 50000, split: str = "train", **kwargs):
        super().__init__(**kwargs)
        assert split in ("train", "test"), "split must be 'train' or 'test'"
        self.size = size
        self.split = split
        self._seed_offset = 0 if split == "train" else TEST_RANDOM_SEED_OFFSET
        filename = WORD_LIST_URL.split("/")[-1]
        word_list_path = download_file_with_lock(WORD_LIST_URL, filename)
        with open(word_list_path, "r", encoding="utf-8") as f:
            self._words = [line.strip() for line in f if line.strip()]

    @property
    def eval_type(self):
        return "generative"

    def num_examples(self):
        return self.size

    def get_example(self, index: int) -> dict:
        seed = self._seed_offset + index
        rng = random.Random(seed)
        gen = _GENERATORS[index % _N_TYPES]
        # Word-based generators take (rng, words); others take just (rng)
        import inspect
        sig = inspect.signature(gen)
        if len(sig.parameters) == 2:
            return gen(rng, self._words)
        else:
            return gen(rng)


# ---------------------------------------------------------------------------
# Preview
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    task = ToolCallingTask(size=12, split="train")
    for i in range(12):
        ex = task.get_example(i)
        print("=" * 80)
        print(f"[Example {i}]")
        print(f"System: {ex['messages'][0]['content'][:120]}...")
        print(f"User:   {ex['messages'][1]['content']}")
        print("Asst:   ", end="")
        for part in ex["messages"][2]["content"]:
            if part["type"] == "text":
                print(part["text"], end="")
            elif part["type"] == "python":
                print(f"<<{part['text']}=", end="")
            elif part["type"] == "python_output":
                print(f"{part['text']}>>", end="")
        print()
