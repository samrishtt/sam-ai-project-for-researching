"""
SAM-AI  ·  Logic Reasoning Task Dataset
=========================================
Provides a curated set of propositional-logic, syllogistic, and
conditional reasoning problems for benchmarking the cognitive pipeline.

Each task is a dictionary with keys:
    id          – unique identifier
    category    – ``propositional`` | ``syllogistic`` | ``conditional`` | ``contrapositive``
    premises    – list of premise strings
    question    – the question to answer
    answer      – ground-truth answer
    difficulty  – 1 (easy) … 5 (hard)
"""

from __future__ import annotations
from typing import Any, Dict, List


def get_logic_tasks() -> List[Dict[str, Any]]:
    """Return the full logic-task benchmark set."""
    return [
        # ── Propositional Logic ──────────────────────────────────
        {
            "id": "LOGIC-001",
            "category": "propositional",
            "premises": [
                "If it rains, the ground is wet.",
                "It is raining.",
            ],
            "question": "Is the ground wet?",
            "answer": True,
            "difficulty": 1,
        },
        {
            "id": "LOGIC-002",
            "category": "propositional",
            "premises": [
                "If it rains, the ground is wet.",
                "The ground is not wet.",
            ],
            "question": "Is it raining?",
            "answer": False,
            "difficulty": 2,
        },
        {
            "id": "LOGIC-003",
            "category": "propositional",
            "premises": [
                "If A then B.",
                "If B then C.",
                "A is true.",
            ],
            "question": "Is C true?",
            "answer": True,
            "difficulty": 2,
        },
        {
            "id": "LOGIC-004",
            "category": "propositional",
            "premises": [
                "If A then B.",
                "If B then C.",
                "C is false.",
            ],
            "question": "Is A true?",
            "answer": False,
            "difficulty": 3,
        },
        {
            "id": "LOGIC-005",
            "category": "propositional",
            "premises": [
                "Either P or Q (or both).",
                "Not P.",
            ],
            "question": "Is Q true?",
            "answer": True,
            "difficulty": 2,
        },

        # ── Syllogistic Logic ───────────────────────────────────
        {
            "id": "LOGIC-006",
            "category": "syllogistic",
            "premises": [
                "All mammals are animals.",
                "All dogs are mammals.",
            ],
            "question": "Are all dogs animals?",
            "answer": True,
            "difficulty": 1,
        },
        {
            "id": "LOGIC-007",
            "category": "syllogistic",
            "premises": [
                "All birds can fly.",
                "Penguins are birds.",
            ],
            "question": "Can penguins fly?",
            "answer": True,  # valid given the premise (even if factually wrong)
            "difficulty": 2,
        },
        {
            "id": "LOGIC-008",
            "category": "syllogistic",
            "premises": [
                "No reptiles are mammals.",
                "All snakes are reptiles.",
            ],
            "question": "Are any snakes mammals?",
            "answer": False,
            "difficulty": 2,
        },
        {
            "id": "LOGIC-009",
            "category": "syllogistic",
            "premises": [
                "Some students are athletes.",
                "All athletes are disciplined.",
            ],
            "question": "Are some students disciplined?",
            "answer": True,
            "difficulty": 3,
        },
        {
            "id": "LOGIC-010",
            "category": "syllogistic",
            "premises": [
                "All roses are flowers.",
                "Some flowers fade quickly.",
            ],
            "question": "Do all roses fade quickly?",
            "answer": False,  # cannot be inferred
            "difficulty": 3,
        },

        # ── Conditional / Contrapositive ─────────────────────────
        {
            "id": "LOGIC-011",
            "category": "contrapositive",
            "premises": [
                "If a number is divisible by 4, it is divisible by 2.",
                "Number X is not divisible by 2.",
            ],
            "question": "Is X divisible by 4?",
            "answer": False,
            "difficulty": 2,
        },
        {
            "id": "LOGIC-012",
            "category": "conditional",
            "premises": [
                "If the alarm sounds, evacuate the building.",
                "If we evacuate, the office is empty.",
                "The alarm sounds.",
            ],
            "question": "Is the office empty?",
            "answer": True,
            "difficulty": 2,
        },
        {
            "id": "LOGIC-013",
            "category": "conditional",
            "premises": [
                "If it is sunny, I will go hiking.",
                "If I go hiking, I will be tired.",
                "If I am tired, I will sleep early.",
                "It is sunny.",
            ],
            "question": "Will I sleep early?",
            "answer": True,
            "difficulty": 3,
        },
        {
            "id": "LOGIC-014",
            "category": "conditional",
            "premises": [
                "If an animal is a cat, it has whiskers.",
                "If an animal has whiskers, it can sense vibrations.",
                "Animal Y does not sense vibrations.",
            ],
            "question": "Is Y a cat?",
            "answer": False,
            "difficulty": 3,
        },
        {
            "id": "LOGIC-015",
            "category": "propositional",
            "premises": [
                "P implies Q.",
                "Q implies R.",
                "R implies S.",
                "Not S.",
            ],
            "question": "Is P true?",
            "answer": False,
            "difficulty": 4,
        },
    ]
