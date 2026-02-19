"""
SAM-AI  ·  Adversarial Test Dataset
======================================
A curated collection of adversarial reasoning tasks designed to stress-test
the SAM-AI cognitive pipeline.

Categories:
    1. Logical Fallacy Traps      — tasks that exploit common reasoning errors
    2. Contradictory Premises     — tasks with internally inconsistent premises
    3. Ambiguous Reasoning        — tasks with multiple valid interpretations

Each task follows the standard SAM-AI task schema:
    id          – unique identifier
    category    – reasoning category
    premises    – list of premise strings (for logic tasks)
    question    – the question to answer
    answer      – ground-truth answer
    difficulty  – 1 (easy) … 5 (hard)
    adversarial_type – one of: fallacy_trap | contradictory | ambiguous
    description – human-readable explanation of the adversarial challenge
"""

from __future__ import annotations
from typing import Any, Dict, List


def get_adversarial_tasks() -> List[Dict[str, Any]]:
    """Return the full adversarial task benchmark set."""
    return (
        _get_fallacy_traps()
        + _get_contradictory_premises()
        + _get_ambiguous_problems()
    )


# ══════════════════════════════════════════════════════════════════════════════
#  1. Logical Fallacy Traps
# ══════════════════════════════════════════════════════════════════════════════

def _get_fallacy_traps() -> List[Dict[str, Any]]:
    """Tasks that exploit common logical fallacies."""
    return [
        # ── Affirming the Consequent ─────────────────────────────────────────
        {
            "id":               "ADV-FALL-001",
            "category":         "propositional",
            "adversarial_type": "fallacy_trap",
            "description":      "Affirming the consequent: P→Q, Q is true does NOT imply P is true.",
            "premises": [
                "If it rains, the ground is wet.",
                "The ground is wet.",
            ],
            "question":   "Is it raining?",
            "answer":     None,   # Cannot be determined — classic fallacy
            "difficulty": 3,
        },
        # ── Denying the Antecedent ───────────────────────────────────────────
        {
            "id":               "ADV-FALL-002",
            "category":         "propositional",
            "adversarial_type": "fallacy_trap",
            "description":      "Denying the antecedent: P→Q, ¬P does NOT imply ¬Q.",
            "premises": [
                "If you study hard, you will pass.",
                "You did not study hard.",
            ],
            "question":   "Will you fail?",
            "answer":     None,   # Cannot be determined
            "difficulty": 3,
        },
        # ── Circular Reasoning ───────────────────────────────────────────────
        {
            "id":               "ADV-FALL-003",
            "category":         "propositional",
            "adversarial_type": "fallacy_trap",
            "description":      "Circular reasoning: conclusion restates a premise.",
            "premises": [
                "The Bible is true because it says so.",
                "The Bible says it is the word of God.",
            ],
            "question":   "Is the Bible the word of God?",
            "answer":     None,   # Circular — cannot be determined from premises alone
            "difficulty": 4,
        },
        # ── Undistributed Middle ─────────────────────────────────────────────
        {
            "id":               "ADV-FALL-004",
            "category":         "syllogistic",
            "adversarial_type": "fallacy_trap",
            "description":      "Undistributed middle: All A are C, All B are C does NOT mean All A are B.",
            "premises": [
                "All cats are animals.",
                "All dogs are animals.",
            ],
            "question":   "Are all cats dogs?",
            "answer":     False,
            "difficulty": 3,
        },
        # ── False Dilemma ────────────────────────────────────────────────────
        {
            "id":               "ADV-FALL-005",
            "category":         "propositional",
            "adversarial_type": "fallacy_trap",
            "description":      "False dilemma: presents only two options when more exist.",
            "premises": [
                "Either you are with us or against us.",
                "You are not with us.",
            ],
            "question":   "Are you against us?",
            "answer":     True,   # Valid given the binary premise (even if premise is fallacious)
            "difficulty": 2,
        },
        # ── Hasty Generalisation ─────────────────────────────────────────────
        {
            "id":               "ADV-FALL-006",
            "category":         "syllogistic",
            "adversarial_type": "fallacy_trap",
            "description":      "Hasty generalisation: 'some' does not imply 'all'.",
            "premises": [
                "Some politicians are corrupt.",
                "Alice is a politician.",
            ],
            "question":   "Is Alice corrupt?",
            "answer":     None,   # Cannot be determined — hasty generalisation
            "difficulty": 3,
        },
        # ── Slippery Slope ───────────────────────────────────────────────────
        {
            "id":               "ADV-FALL-007",
            "category":         "conditional",
            "adversarial_type": "fallacy_trap",
            "description":      "Slippery slope: chained conditionals with unsupported leaps.",
            "premises": [
                "If we allow X, then Y will happen.",
                "If Y happens, then Z will happen.",
                "If Z happens, catastrophe follows.",
                "We allow X.",
            ],
            "question":   "Will catastrophe follow?",
            "answer":     True,   # Logically valid given the premises (even if premises are questionable)
            "difficulty": 3,
        },
    ]


# ══════════════════════════════════════════════════════════════════════════════
#  2. Contradictory Premises
# ══════════════════════════════════════════════════════════════════════════════

def _get_contradictory_premises() -> List[Dict[str, Any]]:
    """Tasks with internally inconsistent or contradictory premises."""
    return [
        # ── Direct Contradiction ─────────────────────────────────────────────
        {
            "id":               "ADV-CONT-001",
            "category":         "propositional",
            "adversarial_type": "contradictory",
            "description":      "Direct contradiction: P and ¬P both asserted.",
            "premises": [
                "It is raining.",
                "It is not raining.",
            ],
            "question":   "Is it raining?",
            "answer":     None,   # Contradictory premises — undefined
            "difficulty": 4,
        },
        # ── Syllogistic Contradiction ────────────────────────────────────────
        {
            "id":               "ADV-CONT-002",
            "category":         "syllogistic",
            "adversarial_type": "contradictory",
            "description":      "Contradictory syllogism: All A are B, and No A are B.",
            "premises": [
                "All birds can fly.",
                "No birds can fly.",
                "Tweety is a bird.",
            ],
            "question":   "Can Tweety fly?",
            "answer":     None,   # Contradictory premises
            "difficulty": 4,
        },
        # ── Implicit Contradiction via Chain ─────────────────────────────────
        {
            "id":               "ADV-CONT-003",
            "category":         "conditional",
            "adversarial_type": "contradictory",
            "description":      "Implicit contradiction via implication chain: P→Q, Q→¬P, P.",
            "premises": [
                "If the light is on, the room is bright.",
                "If the room is bright, the light is off.",
                "The light is on.",
            ],
            "question":   "Is the room bright?",
            "answer":     True,   # First inference fires; contradiction detected by meta-evaluator
            "difficulty": 5,
        },
        # ── Quantifier Contradiction ─────────────────────────────────────────
        {
            "id":               "ADV-CONT-004",
            "category":         "syllogistic",
            "adversarial_type": "contradictory",
            "description":      "Quantifier contradiction: All A are B, Some A are not B.",
            "premises": [
                "All students passed the exam.",
                "Some students did not pass the exam.",
            ],
            "question":   "Did all students pass?",
            "answer":     None,   # Contradictory premises
            "difficulty": 4,
        },
        # ── Numerical Contradiction ──────────────────────────────────────────
        {
            "id":               "ADV-CONT-005",
            "category":         "arithmetic",
            "adversarial_type": "contradictory",
            "description":      "Numerical contradiction: two incompatible values for the same variable.",
            "premises": [],
            "question":   "If x = 5 and x = 7, what is x + 3?",
            "answer":     None,   # Contradictory — x cannot be both 5 and 7
            "difficulty": 3,
        },
    ]


# ══════════════════════════════════════════════════════════════════════════════
#  3. Ambiguous Reasoning Problems
# ══════════════════════════════════════════════════════════════════════════════

def _get_ambiguous_problems() -> List[Dict[str, Any]]:
    """Tasks with genuine ambiguity in interpretation."""
    return [
        # ── Scope Ambiguity ──────────────────────────────────────────────────
        {
            "id":               "ADV-AMB-001",
            "category":         "propositional",
            "adversarial_type": "ambiguous",
            "description":      "Scope ambiguity: 'not A or B' could mean '(not A) or B' or 'not (A or B)'.",
            "premises": [
                "Not A or B is true.",
            ],
            "question":   "Is A true?",
            "answer":     None,   # Ambiguous without parentheses
            "difficulty": 4,
        },
        # ── Referential Ambiguity ────────────────────────────────────────────
        {
            "id":               "ADV-AMB-002",
            "category":         "syllogistic",
            "adversarial_type": "ambiguous",
            "description":      "Referential ambiguity: 'they' is ambiguous.",
            "premises": [
                "John told Mark that he had made a mistake.",
            ],
            "question":   "Who made the mistake?",
            "answer":     None,   # Ambiguous pronoun reference
            "difficulty": 3,
        },
        # ── Temporal Ambiguity ───────────────────────────────────────────────
        {
            "id":               "ADV-AMB-003",
            "category":         "conditional",
            "adversarial_type": "ambiguous",
            "description":      "Temporal ambiguity: 'after' could mean immediately or eventually.",
            "premises": [
                "After the rain stops, the sun will shine.",
                "The rain stopped.",
            ],
            "question":   "Is the sun shining now?",
            "answer":     True,   # Best inference given premises, though timing is ambiguous
            "difficulty": 3,
        },
        # ── Numeric Ambiguity ────────────────────────────────────────────────
        {
            "id":               "ADV-AMB-004",
            "category":         "sequence",
            "adversarial_type": "ambiguous",
            "description":      "Sequence with multiple valid continuations: 1, 2, 4, 8, __",
            "premises": [],
            "question":   "What is the next number in the sequence: 1, 2, 4, 8, __?",
            "answer":     16,     # Most common answer (geometric), but 15 (differences) also valid
            "difficulty": 2,
        },
        # ── Conditional Ambiguity ────────────────────────────────────────────
        {
            "id":               "ADV-AMB-005",
            "category":         "propositional",
            "adversarial_type": "ambiguous",
            "description":      "Biconditional vs conditional ambiguity: 'if and only if' vs 'if'.",
            "premises": [
                "You get a reward if you work hard.",
                "You did not work hard.",
            ],
            "question":   "Do you get a reward?",
            "answer":     None,   # Ambiguous: conditional does not exclude other paths to reward
            "difficulty": 3,
        },
        # ── Vague Quantifier ─────────────────────────────────────────────────
        {
            "id":               "ADV-AMB-006",
            "category":         "syllogistic",
            "adversarial_type": "ambiguous",
            "description":      "Vague quantifier: 'most' is not 'all' — cannot conclude universally.",
            "premises": [
                "Most scientists believe in climate change.",
                "Dr. Smith is a scientist.",
            ],
            "question":   "Does Dr. Smith believe in climate change?",
            "answer":     None,   # Cannot determine from 'most'
            "difficulty": 3,
        },
        # ── Self-Reference Paradox ───────────────────────────────────────────
        {
            "id":               "ADV-AMB-007",
            "category":         "propositional",
            "adversarial_type": "ambiguous",
            "description":      "Self-referential paradox: Liar's paradox variant.",
            "premises": [
                "This statement is false.",
            ],
            "question":   "Is the statement true?",
            "answer":     None,   # Paradox — neither true nor false
            "difficulty": 5,
        },
    ]


# ── Integration helper ────────────────────────────────────────────────────────
def get_adversarial_tasks_by_type(adversarial_type: str) -> List[Dict[str, Any]]:
    """
    Filter adversarial tasks by type.

    Parameters
    ----------
    adversarial_type : str
        One of: 'fallacy_trap', 'contradictory', 'ambiguous'
    """
    return [
        t for t in get_adversarial_tasks()
        if t.get("adversarial_type") == adversarial_type
    ]


def get_adversarial_summary() -> Dict[str, int]:
    """Return a count of tasks per adversarial type."""
    tasks = get_adversarial_tasks()
    summary: Dict[str, int] = {}
    for t in tasks:
        atype = t.get("adversarial_type", "unknown")
        summary[atype] = summary.get(atype, 0) + 1
    return summary
