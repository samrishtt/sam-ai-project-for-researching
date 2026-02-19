"""
SAM-AI  ·  NLP Preprocessing Module
======================================
Converts natural language problem descriptions into the structured task
dictionary format expected by the ReasoningEngine.

Design principles:
- Rule-based parsing only (no external APIs or ML models)
- Handles logic, math, and pattern problem types
- Graceful fallback for unrecognised input

Output task format:
    {
        "id":         str,
        "category":   str,   # propositional | syllogistic | conditional |
                              # contrapositive | arithmetic | algebra |
                              # number_theory | word_problem |
                              # sequence | matrix | analogy
        "premises":   list,  # for logic tasks
        "question":   str,
        "answer":     None,  # unknown at parse time
        "difficulty": int,
        "raw_input":  str,
    }
"""

from __future__ import annotations

import re
import uuid
from typing import Any, Dict, List, Optional, Tuple


# ══════════════════════════════════════════════════════════════════════════════
#  Pattern libraries
# ══════════════════════════════════════════════════════════════════════════════

# Sentence splitters (period, semicolon, newline — but not decimal points)
_SENT_SPLIT = re.compile(r"(?<!\d)\.(?!\d)|;|\n")

# Logic keyword patterns
_IF_THEN        = re.compile(r"\bif\b.+\bthen\b|\bif\b.+,", re.IGNORECASE)
_ALL_SOME_NO    = re.compile(r"^\s*(all|some|no)\s+", re.IGNORECASE)
_IMPLIES        = re.compile(r"\bimplies\b|\bimplication\b", re.IGNORECASE)
_CONTRAPOSITIVE = re.compile(r"\bcontrapositive\b|\bnot\b.+\bdivisible\b", re.IGNORECASE)
_EITHER_OR      = re.compile(r"\beither\b|\bor\b", re.IGNORECASE)

# Math keyword patterns
_ARITHMETIC_OPS = re.compile(r"[\+\-\*\/\^]|\bplus\b|\bminus\b|\btimes\b|\bdivided\b", re.IGNORECASE)
_ALGEBRA_X      = re.compile(r"\bx\s*[\+\-\*\/=]|\bsolve\b|\bequation\b|\bfind\s+x\b|\bwhat\s+is\s+x\b", re.IGNORECASE)
_WORD_PROBLEM   = re.compile(r"\bhow\s+many\b|\bhow\s+much\b|\btotal\b|\bremaining\b|\bleft\b|\bspend\b|\bbuy\b|\bsell\b", re.IGNORECASE)
_NUMBER_THEORY  = re.compile(r"\bprime\b|\bdivisible\b|\bfactor\b|\bgcd\b|\blcm\b|\bmodulo\b|\bremainder\b", re.IGNORECASE)

# Pattern keyword patterns
_SEQUENCE       = re.compile(r"\bnext\s+number\b|\bsequence\b|\bpattern\b|\b__\b|\b\?\b", re.IGNORECASE)
_ANALOGY        = re.compile(r"\bis\s+to\b|\b::\b|\banalogy\b|\bn\s*[→\-]\s*n", re.IGNORECASE)
_MATRIX         = re.compile(r"\bmatrix\b|\bgrid\b|\brow\b|\bcolumn\b|\b\[\d", re.IGNORECASE)

# Question sentence detector
_QUESTION_RE    = re.compile(r"\?$|\bwhat\b|\bwho\b|\bwhere\b|\bwhen\b|\bhow\b|\bwill\b|\bcan\b|\bare\b|\bis\b|\bdoes\b", re.IGNORECASE)

# Number extractor
_NUMBERS        = re.compile(r"-?\d+\.?\d*")


# ══════════════════════════════════════════════════════════════════════════════
#  NLPParser
# ══════════════════════════════════════════════════════════════════════════════

class NLPParser:
    """
    Rule-based natural language → structured task converter.

    Usage
    -----
    >>> parser = NLPParser()
    >>> task = parser.parse("If it rains, the ground is wet. It is raining. Is the ground wet?")
    >>> task["category"]
    'propositional'
    """

    def parse(self, text: str) -> Dict[str, Any]:
        """
        Parse a natural language problem into a structured task dict.

        Parameters
        ----------
        text : str
            Raw natural language problem description.

        Returns
        -------
        dict
            Structured task dictionary compatible with ReasoningEngine.
        """
        text = text.strip()
        sentences = self._split_sentences(text)
        premises, question = self._split_premises_question(sentences)
        category = self._classify(text, premises, question)

        task: Dict[str, Any] = {
            "id":        f"NLP-{uuid.uuid4().hex[:8].upper()}",
            "category":  category,
            "question":  question,
            "answer":    None,
            "difficulty": self._estimate_difficulty(text, premises),
            "raw_input": text,
        }

        # Logic tasks need a premises list
        if category in ("propositional", "syllogistic", "conditional", "contrapositive"):
            task["premises"] = premises if premises else [text]

        # Pattern tasks may have options extracted
        if category in ("sequence", "analogy", "matrix"):
            options = self._extract_options(text)
            if options:
                task["options"] = options

        return task

    # ── Sentence splitting ────────────────────────────────────────────────────
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into individual sentences."""
        parts = _SENT_SPLIT.split(text)
        return [p.strip() for p in parts if p.strip()]

    # ── Premises vs question separation ──────────────────────────────────────
    def _split_premises_question(self, sentences: List[str]) -> Tuple[List[str], str]:
        """
        Heuristically separate premise sentences from the final question.

        The last sentence that ends with '?' or starts with a question word
        is treated as the question; all others are premises.
        """
        if not sentences:
            return [], ""

        # Find the question sentence (prefer the last one with '?')
        question_idx = None
        for i in range(len(sentences) - 1, -1, -1):
            s = sentences[i]
            if s.endswith("?") or _QUESTION_RE.match(s):
                question_idx = i
                break

        if question_idx is None:
            # No clear question — treat last sentence as question
            question_idx = len(sentences) - 1

        question = sentences[question_idx]
        premises = [s for i, s in enumerate(sentences) if i != question_idx]
        return premises, question

    # ── Category classification ───────────────────────────────────────────────
    def _classify(self, full_text: str, premises: List[str], question: str) -> str:
        """
        Classify the task into one of the 11 supported categories.

        Priority order:
        1. Analogy (very specific pattern)
        2. Sequence (next number / __)
        3. Matrix
        4. Contrapositive
        5. Syllogistic
        6. Conditional / Propositional
        7. Algebra
        8. Number theory
        9. Word problem
        10. Arithmetic
        11. Generic fallback → propositional
        """
        combined = full_text.lower()

        # ── Pattern domain ────────────────────────────────────────────────────
        if _ANALOGY.search(full_text):
            return "analogy"
        if _SEQUENCE.search(full_text) and _NUMBERS.search(full_text):
            return "sequence"
        if _MATRIX.search(full_text):
            return "matrix"

        # ── Logic domain ──────────────────────────────────────────────────────
        if _CONTRAPOSITIVE.search(full_text):
            return "contrapositive"

        has_all_some_no = any(_ALL_SOME_NO.match(p) for p in premises)
        if has_all_some_no:
            return "syllogistic"

        has_if_then  = any(_IF_THEN.search(p) for p in premises) or _IF_THEN.search(question)
        has_implies  = _IMPLIES.search(full_text)
        has_either   = _EITHER_OR.search(full_text)
        if has_if_then or has_implies:
            # Multi-step conditional vs simple propositional
            if len(premises) >= 3 or "chain" in combined:
                return "conditional"
            return "propositional"
        if has_either:
            return "propositional"

        # ── Math domain ───────────────────────────────────────────────────────
        if _ALGEBRA_X.search(full_text):
            return "algebra"
        if _NUMBER_THEORY.search(full_text):
            return "number_theory"
        if _WORD_PROBLEM.search(full_text):
            return "word_problem"
        if _ARITHMETIC_OPS.search(full_text) and _NUMBERS.search(full_text):
            return "arithmetic"

        # ── Fallback ──────────────────────────────────────────────────────────
        # If it looks like a yes/no question with premises, treat as propositional
        if premises and (question.endswith("?") or question.lower().startswith(("is ", "are ", "will ", "can ", "does "))):
            return "propositional"

        return "propositional"

    # ── Difficulty estimation ─────────────────────────────────────────────────
    def _estimate_difficulty(self, text: str, premises: List[str]) -> int:
        """
        Heuristically estimate task difficulty on a 1–5 scale.

        Factors:
        - Number of premises (more → harder)
        - Presence of negation
        - Presence of chained conditionals
        - Length of text
        """
        score = 1
        n_premises = len(premises)
        if n_premises >= 4:
            score += 2
        elif n_premises >= 2:
            score += 1

        if re.search(r"\bnot\b|\bno\b|\bnever\b|\bnone\b", text, re.IGNORECASE):
            score += 1
        if re.search(r"\bif\b.+\bif\b|\bchain\b|\btransitive\b", text, re.IGNORECASE):
            score += 1

        return min(5, max(1, score))

    # ── Option extraction ─────────────────────────────────────────────────────
    def _extract_options(self, text: str) -> Optional[List[int]]:
        """
        Extract multiple-choice options from text like '(a) 4 (b) 8 (c) 16'.
        Returns None if no options found.
        """
        # Pattern: (a) 4 (b) 8 ...  or  A. 4  B. 8 ...
        opts = re.findall(r"\([a-dA-D]\)\s*(-?\d+)|[A-D]\.\s*(-?\d+)", text)
        if opts:
            nums = []
            for a, b in opts:
                val = a or b
                if val:
                    nums.append(int(val))
            return nums if nums else None
        return None


# ── Convenience function ──────────────────────────────────────────────────────
def parse_problem(text: str) -> Dict[str, Any]:
    """Module-level convenience wrapper around NLPParser.parse()."""
    return NLPParser().parse(text)
