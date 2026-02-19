"""
SAM-AI  ·  Pattern Recognition Task Dataset
=============================================
Provides sequence-completion, matrix-pattern, and analogical
reasoning problems for testing abstract pattern detection.

Each task is a dictionary with keys:
    id          – unique identifier
    category    – ``sequence`` | ``matrix`` | ``analogy``
    question    – problem statement
    options     – list of candidate answers (if multiple-choice)
    answer      – ground-truth answer
    difficulty  – 1 (easy) … 5 (hard)
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional


def get_pattern_tasks() -> List[Dict[str, Any]]:
    """Return the full pattern-recognition benchmark set."""
    return [
        # ── Numeric Sequences ───────────────────────────────────
        {
            "id": "PAT-001",
            "category": "sequence",
            "question": "What comes next? 2, 4, 6, 8, __",
            "options": None,
            "answer": 10,
            "difficulty": 1,
        },
        {
            "id": "PAT-002",
            "category": "sequence",
            "question": "What comes next? 1, 1, 2, 3, 5, 8, __",
            "options": None,
            "answer": 13,
            "difficulty": 1,
        },
        {
            "id": "PAT-003",
            "category": "sequence",
            "question": "What comes next? 3, 6, 12, 24, __",
            "options": None,
            "answer": 48,
            "difficulty": 2,
        },
        {
            "id": "PAT-004",
            "category": "sequence",
            "question": "What comes next? 1, 4, 9, 16, 25, __",
            "options": None,
            "answer": 36,
            "difficulty": 2,
        },
        {
            "id": "PAT-005",
            "category": "sequence",
            "question": "What comes next? 2, 6, 12, 20, 30, __",
            "options": None,
            "answer": 42,
            "difficulty": 3,
        },
        {
            "id": "PAT-006",
            "category": "sequence",
            "question": "What comes next? 1, 3, 7, 15, 31, __",
            "options": None,
            "answer": 63,
            "difficulty": 3,
        },
        {
            "id": "PAT-007",
            "category": "sequence",
            "question": "What comes next? 0, 1, 1, 2, 3, 5, 8, 13, 21, __",
            "options": None,
            "answer": 34,
            "difficulty": 2,
        },

        # ── Matrix / Grid Patterns ──────────────────────────────
        {
            "id": "PAT-008",
            "category": "matrix",
            "question": (
                "In a 3×3 grid, rows contain [1,2,3], [4,5,6], [7,8,?]. "
                "What replaces '?'?"
            ),
            "options": [7, 8, 9, 10],
            "answer": 9,
            "difficulty": 1,
        },
        {
            "id": "PAT-009",
            "category": "matrix",
            "question": (
                "Each row sums to 15: [2,7,6], [9,5,1], [4,3,?]. "
                "What replaces '?'?"
            ),
            "options": [5, 6, 7, 8],
            "answer": 8,
            "difficulty": 2,
        },
        {
            "id": "PAT-010",
            "category": "matrix",
            "question": (
                "Grid pattern: [1,2,4], [8,16,32], [64,128,?]. "
                "What replaces '?'?"
            ),
            "options": [192, 200, 256, 512],
            "answer": 256,
            "difficulty": 3,
        },

        # ── Analogies ──────────────────────────────────────────
        {
            "id": "PAT-011",
            "category": "analogy",
            "question": "2 is to 4 as 5 is to __",
            "options": [7, 10, 15, 25],
            "answer": 10,
            "difficulty": 1,
        },
        {
            "id": "PAT-012",
            "category": "analogy",
            "question": "3 is to 9 as 7 is to __",
            "options": [14, 21, 49, 56],
            "answer": 49,
            "difficulty": 2,
        },
        {
            "id": "PAT-013",
            "category": "analogy",
            "question": "1 is to 1 as 4 is to 64. The pattern is n→n³. 3 is to __",
            "options": [9, 12, 27, 81],
            "answer": 27,
            "difficulty": 2,
        },
        {
            "id": "PAT-014",
            "category": "analogy",
            "question": (
                "If 'AB' maps to 3, 'CD' maps to 7, then 'EF' maps to __ "
                "(sum of letter positions: A=1,B=2,...)"
            ),
            "options": [9, 11, 13, 15],
            "answer": 11,
            "difficulty": 3,
        },
        {
            "id": "PAT-015",
            "category": "sequence",
            "question": "What comes next? 1, 2, 6, 24, 120, __",
            "options": None,
            "answer": 720,
            "difficulty": 3,
        },
    ]
