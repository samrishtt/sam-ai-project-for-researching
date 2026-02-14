"""
SAM-AI  ·  Mathematical Reasoning Task Dataset
================================================
Provides arithmetic, algebraic, number-theory, and multi-step
word-problem tasks for benchmarking the reasoning engine.

Each task is a dictionary with keys:
    id          – unique identifier
    category    – ``arithmetic`` | ``algebra`` | ``number_theory`` | ``word_problem``
    question    – natural-language problem statement
    answer      – ground-truth numeric answer
    steps_hint  – expected number of reasoning steps
    difficulty  – 1 (easy) … 5 (hard)
"""

from __future__ import annotations
from typing import Any, Dict, List


def get_math_tasks() -> List[Dict[str, Any]]:
    """Return the full math-reasoning benchmark set."""
    return [
        # ── Arithmetic ──────────────────────────────────────────
        {
            "id": "MATH-001",
            "category": "arithmetic",
            "question": "What is 17 × 23?",
            "answer": 391,
            "steps_hint": 2,
            "difficulty": 1,
        },
        {
            "id": "MATH-002",
            "category": "arithmetic",
            "question": "Calculate 144 ÷ 12 + 7 × 3.",
            "answer": 33,
            "steps_hint": 3,
            "difficulty": 2,
        },
        {
            "id": "MATH-003",
            "category": "arithmetic",
            "question": "What is the sum of all integers from 1 to 100?",
            "answer": 5050,
            "steps_hint": 2,
            "difficulty": 2,
        },

        # ── Algebra ─────────────────────────────────────────────
        {
            "id": "MATH-004",
            "category": "algebra",
            "question": "Solve for x: 3x + 7 = 22.",
            "answer": 5,
            "steps_hint": 2,
            "difficulty": 1,
        },
        {
            "id": "MATH-005",
            "category": "algebra",
            "question": "Solve for x: 2x² - 8 = 0. Give the positive root.",
            "answer": 2,
            "steps_hint": 3,
            "difficulty": 2,
        },
        {
            "id": "MATH-006",
            "category": "algebra",
            "question": "If f(x) = x² + 3x + 2, what is f(4)?",
            "answer": 30,
            "steps_hint": 2,
            "difficulty": 2,
        },
        {
            "id": "MATH-007",
            "category": "algebra",
            "question": "Solve: x + y = 10 and x - y = 4. What is x?",
            "answer": 7,
            "steps_hint": 3,
            "difficulty": 3,
        },

        # ── Number Theory ───────────────────────────────────────
        {
            "id": "MATH-008",
            "category": "number_theory",
            "question": "What is the GCD of 48 and 18?",
            "answer": 6,
            "steps_hint": 3,
            "difficulty": 2,
        },
        {
            "id": "MATH-009",
            "category": "number_theory",
            "question": "Is 97 a prime number? Answer 1 for yes, 0 for no.",
            "answer": 1,
            "steps_hint": 3,
            "difficulty": 2,
        },
        {
            "id": "MATH-010",
            "category": "number_theory",
            "question": "What is the LCM of 12 and 15?",
            "answer": 60,
            "steps_hint": 3,
            "difficulty": 2,
        },
        {
            "id": "MATH-011",
            "category": "number_theory",
            "question": "How many prime numbers are there between 1 and 30?",
            "answer": 10,
            "steps_hint": 4,
            "difficulty": 3,
        },

        # ── Word Problems ───────────────────────────────────────
        {
            "id": "MATH-012",
            "category": "word_problem",
            "question": (
                "A train travels at 60 km/h for 2.5 hours. "
                "How many kilometres does it cover?"
            ),
            "answer": 150,
            "steps_hint": 2,
            "difficulty": 1,
        },
        {
            "id": "MATH-013",
            "category": "word_problem",
            "question": (
                "A store sells apples at $2 each and oranges at $3 each. "
                "If you buy 4 apples and 5 oranges, what is the total cost?"
            ),
            "answer": 23,
            "steps_hint": 3,
            "difficulty": 2,
        },
        {
            "id": "MATH-014",
            "category": "word_problem",
            "question": (
                "If a rectangle has a perimeter of 30 cm and one side is 8 cm, "
                "what is the area in cm²?"
            ),
            "answer": 56,
            "steps_hint": 3,
            "difficulty": 2,
        },
        {
            "id": "MATH-015",
            "category": "word_problem",
            "question": (
                "Two pipes fill a tank. Pipe A fills it in 6 hours, "
                "Pipe B fills it in 4 hours. Working together, how many "
                "hours (as a decimal) to fill the tank?"
            ),
            "answer": 2.4,
            "steps_hint": 4,
            "difficulty": 3,
        },
    ]
