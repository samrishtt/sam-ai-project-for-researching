"""
SAM-AI  ·  Scoring & Metrics Utilities
========================================
Implements evaluation primitives used by the meta-evaluator and
performance analytics modules.

Metrics
-------
- Exact-match accuracy
- Partial-credit scoring for structured answers
- Expected Calibration Error (ECE)
- Composite Cognitive Performance Score (CCPS)
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple


# ─── Exact-Match Accuracy ────────────────────────────────────────
def accuracy(predictions: Sequence[Any], ground_truths: Sequence[Any]) -> float:
    """Fraction of *predictions* that exactly equal *ground_truths*."""
    if len(predictions) == 0:
        return 0.0
    correct = sum(1 for p, g in zip(predictions, ground_truths) if p == g)
    return correct / len(predictions)


# ─── Partial-Credit Scoring ─────────────────────────────────────
def partial_credit_score(
    prediction: Any,
    ground_truth: Any,
    reasoning_steps_valid: int = 0,
    total_reasoning_steps: int = 1,
) -> float:
    """Award credit proportional to valid reasoning even when the
    final answer is wrong.

    Returns a value in [0, 1].
    """
    if prediction == ground_truth:
        return 1.0
    # Partial credit = fraction of valid reasoning steps × 0.5
    if total_reasoning_steps == 0:
        return 0.0
    return 0.5 * (reasoning_steps_valid / total_reasoning_steps)


# ─── Expected Calibration Error (ECE) ───────────────────────────
def expected_calibration_error(
    confidences: Sequence[float],
    correctness: Sequence[bool],
    n_bins: int = 10,
) -> float:
    """Estimate ECE using equal-width binning.

    Parameters
    ----------
    confidences : sequence of float
        Model confidence for each prediction.
    correctness : sequence of bool
        Whether each prediction was correct.
    n_bins : int
        Number of equal-width bins.

    Returns
    -------
    float
        Weighted-average absolute calibration gap.
    """
    if len(confidences) == 0:
        return 0.0

    bin_boundaries = [i / n_bins for i in range(n_bins + 1)]
    ece = 0.0
    total = len(confidences)

    for b in range(n_bins):
        lo, hi = bin_boundaries[b], bin_boundaries[b + 1]
        # Gather samples in this bin
        indices = [
            i for i, c in enumerate(confidences)
            if (lo <= c < hi) or (b == n_bins - 1 and c == hi)
        ]
        if not indices:
            continue
        bin_conf = sum(confidences[i] for i in indices) / len(indices)
        bin_acc = sum(1 for i in indices if correctness[i]) / len(indices)
        ece += (len(indices) / total) * abs(bin_acc - bin_conf)

    return ece


# ─── Composite Cognitive Performance Score ───────────────────────
def cognitive_performance_score(
    accuracy_val: float,
    calibration_error: float,
    avg_correction_gain: float = 0.0,
    reasoning_depth_ratio: float = 1.0,
) -> float:
    """Composite metric combining accuracy, calibration, self-correction
    effectiveness, and reasoning depth.

    CCPS = w1·Acc + w2·(1-ECE) + w3·CorrGain + w4·DepthRatio

    All components normalised to [0, 1]; weights sum to 1.
    """
    w1, w2, w3, w4 = 0.40, 0.25, 0.20, 0.15
    score = (
        w1 * accuracy_val
        + w2 * (1.0 - calibration_error)
        + w3 * max(0.0, min(1.0, avg_correction_gain))
        + w4 * max(0.0, min(1.0, reasoning_depth_ratio))
    )
    return round(score, 4)


# ─── Reasoning-Chain Consistency Score ───────────────────────────
def chain_consistency_score(
    step_validities: Sequence[bool],
    step_confidences: Sequence[float],
) -> float:
    """Weighted consistency metric for a reasoning trace.

    Invalid steps with high confidence are penalised more heavily
    (over-confident errors are worse than low-confidence errors).
    """
    if len(step_validities) == 0:
        return 1.0

    score = 0.0
    for valid, conf in zip(step_validities, step_confidences):
        if valid:
            score += conf   # reward: valid & confident
        else:
            score -= conf   # penalty: invalid & confident
    # Normalise to [0, 1]
    normalised = (score / len(step_validities) + 1.0) / 2.0
    return round(max(0.0, min(1.0, normalised)), 4)


# ─── Summary Helper ─────────────────────────────────────────────
def compute_summary_metrics(
    predictions: List[Any],
    ground_truths: List[Any],
    confidences: List[float],
    step_validities_list: List[List[bool]],
    step_confidences_list: List[List[float]],
) -> Dict[str, float]:
    """Compute a full summary dictionary for a batch of solved tasks."""
    acc = accuracy(predictions, ground_truths)
    correctness = [p == g for p, g in zip(predictions, ground_truths)]
    ece = expected_calibration_error(confidences, correctness)
    chain_scores = [
        chain_consistency_score(sv, sc)
        for sv, sc in zip(step_validities_list, step_confidences_list)
    ]
    avg_chain = sum(chain_scores) / max(len(chain_scores), 1)
    ccps = cognitive_performance_score(acc, ece)

    return {
        "accuracy": round(acc, 4),
        "ece": round(ece, 4),
        "avg_chain_consistency": round(avg_chain, 4),
        "ccps": round(ccps, 4),
        "n_tasks": len(predictions),
    }
