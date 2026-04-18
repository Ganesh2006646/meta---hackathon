# Copyright (c) 2026. ExecuCode Environment.

"""
Grading Engine for ExecuCode environment.

Calculates a deterministic reward based on three dimensions:
1. Correctness (50%): Does the code pass all test cases?
2. Performance (30%): Does the code use optimal patterns?
3. Code Quality (20%): Is the code readable and well-structured?
"""

import ast
import re
from typing import Any, Dict, List, Tuple

from .tasks import Task
from .utils import safe_exec


def grade_submission(code: str, task: Task) -> Dict[str, Any]:
    """Calculate scores across correctness, performance, and quality.
    
    Returns a dictionary with:
        correctness_score: float (0.0-1.0)
        performance_score: float (0.0-1.0)
        quality_score: float (0.0-1.0)
        total_reward: float (0.0-1.0)
        test_details: list of dicts with test results
        performance_notes: list of strings
        quality_notes: list of strings
    """
    # 1. Correctness (50%)
    test_details = []
    correct_count = 0
    
    for i, test in enumerate(task.test_cases):
        success, actual, error = safe_exec(code, task.function_name, test.input_args)
        
        passed = False
        if success and actual == test.expected_output:
            passed = True
            correct_count += 1
            
        test_details.append({
            "index": i + 1,
            "input": test.input_args[0] if len(test.input_args) == 1 else test.input_args,
            "expected": test.expected_output,
            "actual": actual if success else "Error",
            "passed": passed,
            "error": error
        })
    
    correctness_score = correct_count / len(task.test_cases) if task.test_cases else 0.0

    # 2. Performance (30%)
    performance_score = 0.5  # Base score for correctness
    performance_notes = []
    
    # Check for optimal patterns
    optimal_matches = 0
    for pattern in task.optimal_patterns:
        if re.search(pattern, code):
            optimal_matches += 1
            performance_notes.append(f"Optimal pattern detected: {pattern}")
            
    if task.optimal_patterns:
        performance_score += (optimal_matches / len(task.optimal_patterns)) * 0.5
    else:
        performance_score = 1.0 # No optimization required
        
    # Check for anti-patterns (penalities)
    for pattern in task.anti_patterns:
        if re.search(pattern, code):
            performance_score -= 0.2
            performance_notes.append(f"Inefficient pattern detected: {pattern}")
            
    performance_score = max(0.0, min(1.0, performance_score))
    
    if correctness_score < 0.5:
        performance_score *= correctness_score # Scalar performance by correctness if significantly broken
        performance_notes.append("Performance score reduced due to low correctness.")

    # 3. Code Quality (20%)
    quality_score, quality_notes = _check_code_quality(code)
    
    # Final Weighted Reward
    total_reward = (
        correctness_score * 0.5 +
        performance_score * 0.3 +
        quality_score * 0.2
    )
    
    # Stabilize reward
    total_reward = round(total_reward, 3)
    
    return {
        "correctness_score": correctness_score,
        "performance_score": performance_score,
        "quality_score": quality_score,
        "total_reward": total_reward,
        "test_details": test_details,
        "performance_notes": performance_notes,
        "quality_notes": quality_notes
    }


def _check_code_quality(code: str) -> Tuple[float, List[str]]:
    """Perform static analysis for code quality."""
    score = 1.0
    notes = []
    
    try:
        tree = ast.parse(code)
    except Exception:
        return 0.0, ["Syntax error prevents quality analysis."]

    # 1. Check for Docstring
    has_docstring = False
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Module)):
            if ast.get_docstring(node):
                has_docstring = True
                break
    
    if not has_docstring:
        score -= 0.2
        notes.append("Missing docstring.")
    else:
        notes.append("Docstring present.")

    # 2. Variable Naming (Simple check for single-character names)
    single_char_vars = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            if len(node.id) == 1 and node.id not in ('i', 'j', 'k', 'x', 'y', 'z', '_'):
                single_char_vars.append(node.id)
                
    if single_char_vars:
        score -= 0.1 * min(len(set(single_char_vars)), 3)
        notes.append(f"Generic variable names found: {', '.join(set(single_char_vars))}")
    else:
        notes.append("Good variable naming conventions.")

    # 3. Code Length (Complexity proxy)
    lines = [l for l in code.splitlines() if l.strip() and not l.strip().startswith("#")]
    if len(lines) > 30:
        score -= 0.1
        notes.append(f"Function is relatively long ({len(lines)} lines).")
    elif len(lines) < 5:
        notes.append("Code is concise.")

    # 4. Built-in usage (Encouraging efficient Pythonic code)
    if "for" in code and "range" in code and ("enumerate" in code or "zip" in code):
        notes.append("Good use of Pythonic iteration (enumerate/zip).")

    return max(0.0, min(1.0, score)), notes
