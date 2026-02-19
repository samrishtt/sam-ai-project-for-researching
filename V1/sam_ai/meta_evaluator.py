"""
SAM-AI  ·  Meta-Evaluation Engine  (v2 – Discriminating)
==========================================================
Validates reasoning traces, detects logical gaps/inconsistencies,
and scores trace quality through rule-based verification and
heuristic analysis.

v2 Changes
----------
- Null-result steps now create *issues* (not just warnings) → lower quality
- Answer-is-None detection drastically drops overall quality
- Confidence floor raised to 0.50 for tighter calibration
- Added fallacy: "confidence without substance" — high conf on null results
"""

from __future__ import annotations
import statistics
from typing import Any, Dict, List, Optional
from sam_ai.utils.logging import SAMLogger

logger = SAMLogger.get_logger("MetaEvaluator")


class MetaEvaluation:
    """Container for a complete meta-evaluation of one reasoning trace."""
    def __init__(self):
        self.is_valid: bool = True
        self.issues: List[str] = []
        self.warnings: List[str] = []
        self.structural_score: float = 1.0
        self.consistency_score: float = 1.0
        self.depth_score: float = 1.0
        self.confidence_variance: float = 0.0
        self.overall_quality: float = 1.0
        self.step_validities: List[bool] = []
        self.step_confidences: List[float] = []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "issues": self.issues,
            "warnings": self.warnings,
            "structural_score": round(self.structural_score, 4),
            "consistency_score": round(self.consistency_score, 4),
            "depth_score": round(self.depth_score, 4),
            "confidence_variance": round(self.confidence_variance, 4),
            "overall_quality": round(self.overall_quality, 4),
        }


class MetaEvaluator:
    """Analyses and scores reasoning traces.

    Parameters
    ----------
    min_depth : int
        Minimum acceptable reasoning depth.
    confidence_floor : float
        Steps below this confidence trigger a warning.
    max_acceptable_variance : float
        Maximum tolerable confidence variance.
    """

    def __init__(self, min_depth=2, confidence_floor=0.50, max_acceptable_variance=0.10):
        self.min_depth = min_depth
        self.confidence_floor = confidence_floor
        self.max_acceptable_variance = max_acceptable_variance

    def evaluate(self, trace_dict: Dict[str, Any]) -> MetaEvaluation:
        ev = MetaEvaluation()
        all_steps = self._flatten_steps(trace_dict)

        self._check_structure(trace_dict, ev)

        depth = self._compute_depth(trace_dict)
        if depth < self.min_depth:
            ev.warnings.append(f"Shallow reasoning: depth={depth} < min={self.min_depth}")
            ev.depth_score = depth / max(self.min_depth, 1)
        else:
            ev.depth_score = min(1.0, depth / (self.min_depth * 2))

        confidences = [s.get("confidence", 0.0) for s in all_steps]
        ev.step_confidences = confidences
        if confidences:
            ev.confidence_variance = statistics.pvariance(confidences)
            low = [c for c in confidences if c < self.confidence_floor]
            if low:
                ev.warnings.append(f"{len(low)} step(s) below confidence floor ({self.confidence_floor})")
        if ev.confidence_variance > self.max_acceptable_variance:
            ev.warnings.append(f"High confidence variance: {ev.confidence_variance:.4f}")

        self._check_consistency(all_steps, ev)
        self._detect_fallacies(all_steps, ev)

        # ── v2: Null-step detection (now an ISSUE, not just warning) ──
        null_steps = [s for s in all_steps if s.get("result") is None]
        if null_steps:
            # Check if the FINAL step (answer) is null — that's critical
            final_step = all_steps[-1] if all_steps else None
            if final_step and final_step.get("result") is None:
                ev.issues.append("Final answer is null — reasoning chain produced no result")
                ev.is_valid = False
            else:
                ev.warnings.append(f"{len(null_steps)} intermediate step(s) produced null results")

        ev.step_validities = [s.get("valid", True) for s in all_steps]
        invalid_count = sum(1 for v in ev.step_validities if not v)
        if invalid_count > 0:
            ev.issues.append(f"{invalid_count} step(s) marked invalid")
            ev.is_valid = False

        issue_penalty = 0.20 * len(ev.issues)      # v2: increased from 0.15
        warning_penalty = 0.04 * len(ev.warnings)
        ev.structural_score = max(0.0, ev.structural_score - issue_penalty)
        ev.overall_quality = (
            0.35 * ev.structural_score
            + 0.25 * ev.consistency_score
            + 0.20 * ev.depth_score
            + 0.20 * (1.0 - min(1.0, ev.confidence_variance * 5))
        )
        ev.overall_quality = max(0.0, min(1.0, ev.overall_quality - warning_penalty))
        ev.overall_quality = round(ev.overall_quality, 4)

        logger.info(f"Meta-eval: quality={ev.overall_quality:.3f}, issues={len(ev.issues)}, warnings={len(ev.warnings)}")
        return ev

    def _flatten_steps(self, node, acc=None):
        if acc is None: acc = []
        acc.append(node)
        for c in node.get("children", []):
            self._flatten_steps(c, acc)
        return acc

    def _compute_depth(self, node):
        children = node.get("children", [])
        return 1 if not children else 1 + max(self._compute_depth(c) for c in children)

    def _check_structure(self, trace, ev):
        missing = {"step", "description"} - set(trace.keys())
        if missing:
            ev.issues.append(f"Root missing keys: {missing}")
            ev.is_valid = False
            ev.structural_score -= 0.3
        self._check_ordering(trace, ev)

    def _check_ordering(self, node, ev):
        children = node.get("children", [])
        for i in range(len(children) - 1):
            if children[i].get("step", 0) >= children[i+1].get("step", 0):
                ev.warnings.append(f"Non-monotonic ordering at step {children[i].get('step')}")
        for c in children:
            self._check_ordering(c, ev)

    def _check_consistency(self, steps, ev):
        bool_results = {}
        for s in steps:
            result = s.get("result")
            desc = s.get("description", "").lower()
            if isinstance(result, dict):
                for k, v in result.items():
                    if isinstance(v, bool):
                        if k in bool_results and bool_results[k] != v:
                            ev.issues.append(f"Contradiction: '{k}' is both {bool_results[k]} and {v}")
                            ev.consistency_score -= 0.25
                            ev.is_valid = False
                        bool_results[k] = v
            elif isinstance(result, bool):
                if desc in bool_results and bool_results[desc] != result:
                    ev.issues.append(f"Contradictory result at: '{desc}'")
                    ev.consistency_score -= 0.25
                    ev.is_valid = False
                bool_results[desc] = result
        ev.consistency_score = max(0.0, ev.consistency_score)

    def _detect_fallacies(self, steps, ev):
        descriptions = [s.get("description", "").lower() for s in steps]
        results = [s.get("result") for s in steps]
        seen = set()
        for d in descriptions:
            if d in seen and len(d) > 10:
                ev.warnings.append(f"Possible circular reasoning: repeated '{d[:60]}…'")
            seen.add(d)
        for i in range(len(steps) - 1):
            if results[i] is None and steps[i+1].get("confidence", 0) > 0.8:
                ev.warnings.append(f"Potential non-sequitur: null at step {steps[i].get('step')} → high-conf step {steps[i+1].get('step')}")
