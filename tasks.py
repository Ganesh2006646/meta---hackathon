"""Task definitions for the ExecuCode environment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TestCase:
    """A single deterministic correctness check."""

    input_args: tuple[Any, ...]
    expected_output: Any


@dataclass(frozen=True)
class Task:
    """Immutable coding task definition."""

    task_id: int
    difficulty: str
    title: str
    description: str
    buggy_code: str
    function_name: str
    test_cases: tuple[TestCase, ...]
    optimal_patterns: tuple[str, ...]
    anti_patterns: tuple[str, ...]
    reference_solution: str
    scoring_weights: tuple[float, float, float]


_TASK_0_BUGGY = '''\
def append_to_history(item, history=[]):
    """Append an item and return the history list."""
    history.append(item)
    return history
'''

_TASK_0_REFERENCE = '''\
def append_to_history(item, history=None):
    """Append an item and return a list without leaking default state."""
    if history is None:
        history = []
    history.append(item)
    return history
'''

TASK_0 = Task(
    task_id=0,
    difficulty="easy",
    title="Python Gotcha: Mutable Default Argument",
    description=(
        "The function `append_to_history(item, history=[])` uses a mutable "
        "default list. That means values can leak across separate calls.\n\n"
        "Fix it by using `None` as the default and creating a fresh list only "
        "when needed.\n\n"
        "```python\n" + _TASK_0_BUGGY + "```"
    ),
    buggy_code=_TASK_0_BUGGY,
    function_name="append_to_history",
    test_cases=(
        TestCase(input_args=("alpha",), expected_output=["alpha"]),
        TestCase(input_args=("beta",), expected_output=["beta"]),
        TestCase(input_args=("gamma", ["seed"]), expected_output=["seed", "gamma"]),
        TestCase(input_args=("delta",), expected_output=["delta"]),
        TestCase(input_args=("epsilon", []), expected_output=["epsilon"]),
        TestCase(input_args=("zeta",), expected_output=["zeta"]),
    ),
    optimal_patterns=(
        r"def\s+append_to_history\s*\([^)]*history\s*=\s*None",
        r"if\s+history\s+is\s+None",
    ),
    anti_patterns=(r"def\s+append_to_history\s*\([^)]*history\s*=\s*\[\s*\]",),
    reference_solution=_TASK_0_REFERENCE,
    scoring_weights=(1.0, 0.0, 0.0),
)


_TASK_1_BUGGY = '''\
def count_paths(grid):
    """Count right/down paths from top-left to bottom-right around blocked cells."""
    if not grid or not grid[0]:
        return 0

    rows = len(grid)
    cols = len(grid[0])

    def dfs(row, col):
        if row >= rows or col >= cols:
            return 0
        if grid[row][col] == 1:
            return 0
        if row == rows - 1 and col == cols - 1:
            return 1
        return dfs(row + 1, col) + dfs(row, col + 1)

    return dfs(0, 0)
'''

_TASK_1_REFERENCE = '''\
def count_paths(grid):
    """Count right/down paths using memoization for O(rows*cols) complexity."""
    if not grid or not grid[0]:
        return 0

    rows = len(grid)
    cols = len(grid[0])
    if grid[0][0] == 1 or grid[rows - 1][cols - 1] == 1:
        return 0

    memo = {}

    def dfs(row, col):
        if row >= rows or col >= cols:
            return 0
        if grid[row][col] == 1:
            return 0
        if row == rows - 1 and col == cols - 1:
            return 1

        key = (row, col)
        if key in memo:
            return memo[key]

        memo[key] = dfs(row + 1, col) + dfs(row, col + 1)
        return memo[key]

    return dfs(0, 0)
'''

TASK_1 = Task(
    task_id=1,
    difficulty="medium",
    title="Dynamic Programming: Grid Path Counting",
    description=(
        "The function `count_paths(grid)` should count how many ways there are "
        "to move from the top-left cell to the bottom-right cell, moving only "
        "right or down, while avoiding blocked cells marked with `1`.\n\n"
        "The current recursive solution recomputes the same subproblems and is "
        "too slow on larger grids.\n\n"
        "Optimize it with memoization (cache overlapping subproblems) so it "
        "runs in O(rows * cols).\n\n"
        "```python\n" + _TASK_1_BUGGY + "```"
    ),
    buggy_code=_TASK_1_BUGGY,
    function_name="count_paths",
    test_cases=(
        TestCase(input_args=([],), expected_output=0),
        TestCase(input_args=([[0]],), expected_output=1),
        TestCase(input_args=([[1]],), expected_output=0),
        TestCase(input_args=([[0, 0], [0, 0]],), expected_output=2),
        TestCase(input_args=([[0, 1], [0, 0]],), expected_output=1),
        TestCase(
            input_args=([[0, 0, 0], [0, 1, 0], [0, 0, 0]],),
            expected_output=2,
        ),
        TestCase(
            input_args=(
                [
                    [0, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 0],
                ],
            ),
            expected_output=4,
        ),
        TestCase(
            input_args=([[0] * 13 for _ in range(13)],),
            expected_output=2704156,
        ),
        TestCase(
            input_args=(
                [
                    [0, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1],
                ],
            ),
            expected_output=0,
        ),
    ),
    optimal_patterns=(
        r"\bmemo\b",
        r"\bcache\b",
        r"@lru_cache",
        r"if\s+\w+\s+in\s+memo",
        r"memo\[\w+\]\s*=",
    ),
    anti_patterns=(
        r"return\s+dfs\(\s*row\s*\+\s*1\s*,\s*col\s*\)\s*\+\s*dfs\(\s*row\s*,\s*col\s*\+\s*1\s*\)",
    ),
    reference_solution=_TASK_1_REFERENCE,
    scoring_weights=(0.75, 0.25, 0.0),
)


_TASK_2_BUGGY = '''\
def chunk_document(text, max_chars):
    chunks = []
    current_chunk = ""
    for char in text:
        current_chunk += char
        if len(current_chunk) >= max_chars:
            chunks.append(current_chunk)
            current_chunk = ""
    if current_chunk:
        chunks.append(current_chunk)
    return chunks
'''

_TASK_2_REFERENCE = '''\
def chunk_document(text: str, max_chars: int) -> list[str]:
    """Split text into chunks up to max_chars, preserving full words."""
    if max_chars <= 0:
        return []

    words = text.split()
    if not words:
        return []

    chunks: list[str] = []
    current_words: list[str] = []
    current_len = 0

    for word in words:
        word_len = len(word)
        if word_len > max_chars:
            if current_words:
                chunks.append(" ".join(current_words))
                current_words = []
                current_len = 0
            chunks.append(word)
            continue

        proposed_len = word_len if not current_words else current_len + 1 + word_len
        if proposed_len <= max_chars:
            current_words.append(word)
            current_len = proposed_len
        else:
            chunks.append(" ".join(current_words))
            current_words = [word]
            current_len = word_len

    if current_words:
        chunks.append(" ".join(current_words))
    return chunks
'''

TASK_2 = Task(
    task_id=2,
    difficulty="hard",
    title="AI Infra: RAG Document Chunker",
    description=(
        "The function `chunk_document(text, max_chars)` should split a long "
        "document into chunks for embedding pipelines.\n\n"
        "Current issues in the buggy version:\n"
        "1. Splits words in half by chunking per character.\n"
        "2. Uses repeated string concatenation in a loop.\n"
        "3. Misses edge cases like extra whitespace and long words.\n"
        "4. Lacks clean documentation and naming.\n\n"
        "Build a fast, readable chunker that preserves word boundaries and "
        "handles tricky inputs deterministically.\n\n"
        "```python\n" + _TASK_2_BUGGY + "```"
    ),
    buggy_code=_TASK_2_BUGGY,
    function_name="chunk_document",
    test_cases=(
        TestCase(input_args=("", 10), expected_output=[]),
        TestCase(input_args=("single", 10), expected_output=["single"]),
        TestCase(
            input_args=("alpha beta gamma", 10),
            expected_output=["alpha beta", "gamma"],
        ),
        TestCase(
            input_args=("  alpha   beta   gamma  ", 10),
            expected_output=["alpha beta", "gamma"],
        ),
        TestCase(
            input_args=("one two three four five", 7),
            expected_output=["one two", "three", "four", "five"],
        ),
        TestCase(
            input_args=("supercalifragilistic tiny words", 8),
            expected_output=["supercalifragilistic", "tiny", "words"],
        ),
        TestCase(
            input_args=("AI systems need robust chunking.", 12),
            expected_output=["AI systems", "need robust", "chunking."],
        ),
        TestCase(input_args=("a b c", 1), expected_output=["a", "b", "c"]),
        TestCase(input_args=("   \n\t  ", 5), expected_output=[]),
    ),
    optimal_patterns=(
        r"\.split\(\)",
        r"\.append\(",
        r"\" \"\.join\(",
        r"if\s+current_words",
    ),
    anti_patterns=(
        r"for\s+\w+\s+in\s+text",
        r"current_chunk\s*\+=",
    ),
    reference_solution=_TASK_2_REFERENCE,
    scoring_weights=(0.55, 0.15, 0.30),
)


ALL_TASKS: tuple[Task, ...] = (TASK_0, TASK_1, TASK_2)


def get_task(task_id: int) -> Task:
    """Return a task by id, wrapping around when needed."""

    return ALL_TASKS[task_id % len(ALL_TASKS)]
