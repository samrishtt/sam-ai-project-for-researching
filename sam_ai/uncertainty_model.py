"""
SAM-AI  ·  Uncertainty Modeling Module
========================================
Assigns probabilistic confidence scores to reasoning steps and
estimates overall reliability using Bayesian-inspired heuristics.

Key capabilities:
- Per-step confidence estimation based on domain, depth, and complexity
- Aggregate trace reliability via confidence propagation
- Calibrated uncertainty quantification with temperature scaling
"""

from __future__ import annotations
import math
from typing import Any, Dict, List, Tuple
from sam_ai.utils.logging import SAMLogger

logger = SAMLogger.get_logger("UncertaintyModel")


class UncertaintyEstimate:
    """Encapsulates uncertainty information for a reasoning result."""
    def __init__(self):
        self.step_confidences: List[float] = []
        self.aggregate_confidence: float = 0.0
        self.reliability_rating: str = "unknown"
        self.entropy: float = 0.0
        self.calibrated_confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_confidences": [round(c, 4) for c in self.step_confidences],
            "aggregate_confidence": round(self.aggregate_confidence, 4),
            "calibrated_confidence": round(self.calibrated_confidence, 4),
            "reliability_rating": self.reliability_rating,
            "entropy": round(self.entropy, 4),
        }


class UncertaintyModel:
    """Probabilistic confidence estimation for reasoning traces.

    Parameters
    ----------
    temperature : float
        Calibration temperature for Platt scaling (>1 = softer).
    depth_decay : float
        Per-level confidence decay factor for deep chains.
    domain_priors : dict
        Prior confidence adjustments per task category.
    """

    _DEFAULT_PRIORS = {
        "propositional": 0.92, "syllogistic": 0.88, "conditional": 0.90,
        "contrapositive": 0.90, "arithmetic": 0.95, "algebra": 0.88,
        "number_theory": 0.85, "word_problem": 0.80,
        "sequence": 0.82, "matrix": 0.78, "analogy": 0.75,
    }

    def __init__(self, temperature=1.2, depth_decay=0.98, domain_priors=None):
        self.temperature = temperature
        self.depth_decay = depth_decay
        self.domain_priors = domain_priors or self._DEFAULT_PRIORS

    def estimate(self, trace_dict: Dict, category: str = "unknown") -> UncertaintyEstimate:
        """Compute uncertainty estimates for a reasoning trace."""
        ue = UncertaintyEstimate()
        all_steps = self._flatten(trace_dict)

        # Per-step confidence with depth decay
        prior = self.domain_priors.get(category, 0.80)
        raw_confs = []
        for i, step in enumerate(all_steps):
            c = step.get("confidence", prior)
            depth_factor = self.depth_decay ** i
            adjusted = c * depth_factor
            raw_confs.append(adjusted)

        ue.step_confidences = raw_confs

        # Aggregate: geometric mean (penalises weak links)
        if raw_confs:
            log_sum = sum(math.log(max(c, 1e-10)) for c in raw_confs)
            ue.aggregate_confidence = math.exp(log_sum / len(raw_confs))
        else:
            ue.aggregate_confidence = 0.0

        # Temperature-scaled calibration
        ue.calibrated_confidence = self._calibrate(ue.aggregate_confidence)

        # Shannon entropy of step distribution
        ue.entropy = self._entropy(raw_confs)

        # Reliability rating
        cc = ue.calibrated_confidence
        if cc >= 0.85:
            ue.reliability_rating = "HIGH"
        elif cc >= 0.65:
            ue.reliability_rating = "MODERATE"
        elif cc >= 0.45:
            ue.reliability_rating = "LOW"
        else:
            ue.reliability_rating = "VERY_LOW"

        logger.info(
            f"Uncertainty: aggr={ue.aggregate_confidence:.3f}, "
            f"calib={ue.calibrated_confidence:.3f}, "
            f"rating={ue.reliability_rating}"
        )
        return ue

    def _calibrate(self, p: float) -> float:
        """Platt-style temperature scaling."""
        if p <= 0 or p >= 1:
            return max(0.0, min(1.0, p))
        logit = math.log(p / (1 - p))
        scaled = logit / self.temperature
        return 1.0 / (1.0 + math.exp(-scaled))

    def _entropy(self, confs: List[float]) -> float:
        if not confs:
            return 0.0
        total = sum(confs)
        if total == 0:
            return 0.0
        probs = [c / total for c in confs]
        return -sum(p * math.log2(max(p, 1e-10)) for p in probs)

    def _flatten(self, node, acc=None):
        if acc is None: acc = []
        acc.append(node)
        for c in node.get("children", []):
            self._flatten(c, acc)
        return acc
