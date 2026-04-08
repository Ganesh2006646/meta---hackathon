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
def sum_positive(numbers):
    """Return the sum of all positive numbers in the list."""
    total = 0
    for i in range(len(numbers)):
        total += numbers[i]
    return total
'''

_TASK_0_REFERENCE = '''\
def sum_positive(numbers):
    """Return the sum of all positive numbers in the list."""
    total = 0
    for number in numbers:
        if number > 0:
            total += number
    return total
'''

TASK_0 = Task(
    task_id=0,
    difficulty="easy",
    title="Bug Fix: sum_positive",
    description=(
        "The function `sum_positive(numbers)` should return the sum of all "
        "positive numbers in the input list. The current implementation sums "
        "every number, including negatives and zero.\n\n"
        "Fix the code so that only values greater than zero are included.\n\n"
        "```python\n" + _TASK_0_BUGGY + "```"
    ),
    buggy_code=_TASK_0_BUGGY,
    function_name="sum_positive",
    test_cases=(
        TestCase(input_args=([1, 2, 3],), expected_output=6),
        TestCase(input_args=([5, -3, 10, -1],), expected_output=15),
        TestCase(input_args=([-1, -2, -3],), expected_output=0),
        TestCase(input_args=([],), expected_output=0),
        TestCase(input_args=([0, 0, 0],), expected_output=0),
        TestCase(input_args=([100],), expected_output=100),
        TestCase(input_args=([-5, 3, -2, 7, 0],), expected_output=10),
    ),
    optimal_patterns=(r"if\s+\w+\s*>\s*0", r"if\s+\w+\s*>=\s*1"),
    anti_patterns=(),
    reference_solution=_TASK_0_REFERENCE,
    scoring_weights=(1.0, 0.0, 0.0),
)


_TASK_1_BUGGY = '''\
def find_duplicates(numbers):
    """Return a sorted list of duplicate values in the input list."""
    duplicates = []
    for i in range(len(numbers)):
        for j in range(i + 1, len(numbers)):
            if numbers[i] == numbers[j] and numbers[i] not in duplicates:
                duplicates.append(numbers[i])
    duplicates.sort()
    return duplicates
'''

_TASK_1_REFERENCE = '''\
def find_duplicates(numbers):
    """Return a sorted list of duplicate values in the input list."""
    seen = set()
    duplicates = set()
    for number in numbers:
        if number in seen:
            duplicates.add(number)
        seen.add(number)
    return sorted(duplicates)
'''

TASK_1 = Task(
    task_id=1,
    difficulty="medium",
    title="Optimization: find_duplicates",
    description=(
        "The function `find_duplicates(numbers)` returns a sorted list of "
        "duplicate values, but the current implementation uses O(n^2) nested "
        "loops and repeated list membership checks.\n\n"
        "Optimize it to O(n) or O(n log n) while keeping the output identical.\n\n"
        "```python\n" + _TASK_1_BUGGY + "```"
    ),
    buggy_code=_TASK_1_BUGGY,
    function_name="find_duplicates",
    test_cases=(
        TestCase(input_args=([1, 2, 3, 2, 4, 3],), expected_output=[2, 3]),
        TestCase(input_args=([1, 1, 1],), expected_output=[1]),
        TestCase(input_args=([1, 2, 3],), expected_output=[]),
        TestCase(input_args=([],), expected_output=[]),
        TestCase(input_args=([5, 5, 5, 5],), expected_output=[5]),
        TestCase(input_args=([10, 20, 30, 10, 20],), expected_output=[10, 20]),
        TestCase(input_args=([1],), expected_output=[]),
    ),
    optimal_patterns=(
        r"\bset\s*\(",
        r"\bdict\s*\(",
        r"\bCounter\s*\(",
        r"\bdefaultdict\s*\(",
    ),
    anti_patterns=(
        r"for\s+\w+\s+in\s+range.*:\s*\n\s+for\s+\w+\s+in\s+range",
        r"\bnot\s+in\s+\w+\b.*\.append",
    ),
    reference_solution=_TASK_1_REFERENCE,
    scoring_weights=(0.7, 0.3, 0.0),
)


_TASK_2_BUGGY = '''\
def word_frequency(t):
    d = {}
    w = t.split(" ")
    for x in w:
        if x in d:
            d[x] = d[x] + 1
        else:
            d[x] = 1
    r = []
    for k in d:
        r.append((k, d[k]))
    for i in range(len(r)):
        for j in range(i + 1, len(r)):
            if r[j][1] > r[i][1]:
                r[i], r[j] = r[j], r[i]
    return r
'''

_TASK_2_REFERENCE = '''\
def word_frequency(text):
    """Return (word, count) pairs sorted by descending frequency."""
    import re

    words = re.findall(r"[a-zA-Z]+", text.lower())
    counts = {}
    for word in words:
        counts[word] = counts.get(word, 0) + 1
    return sorted(counts.items(), key=lambda item: item[1], reverse=True)
'''

TASK_2 = Task(
    task_id=2,
    difficulty="hard",
    title="Multi-Objective: word_frequency",
    description=(
        "The function `word_frequency(text)` should return a list of "
        "`(word, count)` tuples sorted by count descending.\n\n"
        "The current implementation has multiple problems:\n"
        "1. It is case-sensitive, so `Hello` and `hello` are counted separately.\n"
        "2. It does not strip punctuation, so `hello,` and `hello` differ.\n"
        "3. It uses O(n^2) bubble sort instead of built-in sorting.\n"
        "4. It uses terse names and has no docstring.\n\n"
        "Fix the bugs, improve performance, and improve readability.\n\n"
        "```python\n" + _TASK_2_BUGGY + "```"
    ),
    buggy_code=_TASK_2_BUGGY,
    function_name="word_frequency",
    test_cases=(
        TestCase(
            input_args=("hello world hello",),
            expected_output=[("hello", 2), ("world", 1)],
        ),
        TestCase(input_args=("Hello hello HELLO",), expected_output=[("hello", 3)]),
        TestCase(
            input_args=("cat, dog. cat! Dog",),
            expected_output=[("cat", 2), ("dog", 2)],
        ),
        TestCase(input_args=("a",), expected_output=[("a", 1)]),
        TestCase(
            input_args=("the the the a a b",),
            expected_output=[("the", 3), ("a", 2), ("b", 1)],
        ),
    ),
    optimal_patterns=(
        r"\.lower\(\)",
        r"\bsorted\s*\(",
        r"re\.findall",
        r"\.strip\(",
    ),
    anti_patterns=(
        r"for\s+\w+\s+in\s+range.*:\s*\n\s+for\s+\w+\s+in\s+range",
    ),
    reference_solution=_TASK_2_REFERENCE,
    scoring_weights=(0.5, 0.3, 0.2),
)


ALL_TASKS: tuple[Task, ...] = (TASK_0, TASK_1, TASK_2)


def get_task(task_id: int) -> Task:
    """Return a task by id, wrapping around when needed."""

    return ALL_TASKS[task_id % len(ALL_TASKS)]
