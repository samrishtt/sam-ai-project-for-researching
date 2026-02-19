"""
SAM-AI  ·  Performance Analytics Module
=========================================
Tracks accuracy trends, monitors confidence calibration,
and detects systematic reasoning weaknesses over time.
"""

from __future__ import annotations
import json, os
from typing import Any, Dict, List, Optional
from sam_ai.utils.scoring import (
    accuracy,
    expected_calibration_error,
    cognitive_performance_score,
    chain_consistency_score,
)
from sam_ai.utils.logging import SAMLogger

logger = SAMLogger.get_logger("PerformanceAnalyzer")


class PerformanceRecord:
    """One snapshot of performance for a batch of tasks."""
    def __init__(self, round_id: int):
        self.round_id = round_id
        self.accuracy: float = 0.0
        self.ece: float = 0.0
        self.ccps: float = 0.0
        self.avg_chain_consistency: float = 0.0
        self.n_tasks: int = 0
        self.per_category: Dict[str, Dict[str, float]] = {}
        self.correction_rate: float = 0.0
        self.avg_confidence: float = 0.0
        self.avg_meta_quality: float = 0.0

    def to_dict(self):
        return {
            "round_id": self.round_id,
            "accuracy": round(self.accuracy, 4),
            "ece": round(self.ece, 4),
            "ccps": round(self.ccps, 4),
            "avg_chain_consistency": round(self.avg_chain_consistency, 4),
            "n_tasks": self.n_tasks,
            "per_category": self.per_category,
            "correction_rate": round(self.correction_rate, 4),
            "avg_confidence": round(self.avg_confidence, 4),
            "avg_meta_quality": round(self.avg_meta_quality, 4),
        }


class PerformanceAnalyzer:
    """Longitudinal performance tracking and weakness detection.

    Parameters
    ----------
    history_path : str or None
        If given, append records as JSON to this file.
    """

    def __init__(self, history_path: Optional[str] = None):
        self.history: List[PerformanceRecord] = []
        self.history_path = history_path

    def record_batch(
        self,
        round_id: int,
        task_results: List[Dict[str, Any]],
    ) -> PerformanceRecord:
        """Compute and store metrics for a batch of task results.

        Each item in *task_results* must contain:
            prediction, ground_truth, confidence,
            step_validities, step_confidences,
            category, was_corrected
        """
        rec = PerformanceRecord(round_id)
        rec.n_tasks = len(task_results)

        preds = [r["prediction"] for r in task_results]
        truths = [r["ground_truth"] for r in task_results]
        confs = [r["confidence"] for r in task_results]
        correctness = [p == g for p, g in zip(preds, truths)]

        rec.accuracy = accuracy(preds, truths)
        rec.ece = expected_calibration_error(confs, correctness)
        rec.avg_confidence = sum(confs) / max(len(confs), 1)

        chain_scores = []
        for r in task_results:
            cs = chain_consistency_score(
                r.get("step_validities", []),
                r.get("step_confidences", []),
            )
            chain_scores.append(cs)
        rec.avg_chain_consistency = sum(chain_scores) / max(len(chain_scores), 1)

        corrected = sum(1 for r in task_results if r.get("was_corrected"))
        rec.correction_rate = corrected / max(rec.n_tasks, 1)

        meta_qualities = [r.get("meta_quality", 0.0) for r in task_results]
        rec.avg_meta_quality = sum(meta_qualities) / max(len(meta_qualities), 1)

        rec.ccps = cognitive_performance_score(
            rec.accuracy, rec.ece,
            avg_correction_gain=rec.correction_rate * 0.5,
            reasoning_depth_ratio=rec.avg_chain_consistency,
        )

        # Per-category breakdown
        cats: Dict[str, List] = {}
        for r, correct in zip(task_results, correctness):
            cat = r.get("category", "unknown")
            cats.setdefault(cat, []).append(correct)
        for cat, vals in cats.items():
            cat_acc = sum(vals) / len(vals)
            rec.per_category[cat] = {"accuracy": round(cat_acc, 4), "n": len(vals)}

        self.history.append(rec)

        # Persist
        if self.history_path:
            os.makedirs(os.path.dirname(self.history_path) or ".", exist_ok=True)
            with open(self.history_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec.to_dict()) + "\n")

        logger.info(f"Round {round_id}: acc={rec.accuracy:.3f}, ece={rec.ece:.3f}, ccps={rec.ccps:.3f}")
        return rec

    def detect_weaknesses(self) -> List[str]:
        """Analyse history to detect systematic weaknesses."""
        findings: List[str] = []
        if len(self.history) < 1:
            return findings

        latest = self.history[-1]

        # Category-level weaknesses
        for cat, info in latest.per_category.items():
            if info["accuracy"] < 0.5 and info["n"] >= 2:
                findings.append(f"Weak category: '{cat}' accuracy={info['accuracy']:.2f} (n={info['n']})")

        # Calibration drift
        if latest.ece > 0.15:
            findings.append(f"Poor calibration: ECE={latest.ece:.3f} (should be < 0.15)")

        # Declining accuracy trend
        if len(self.history) >= 3:
            recent_accs = [h.accuracy for h in self.history[-3:]]
            if all(recent_accs[i] > recent_accs[i+1] for i in range(len(recent_accs)-1)):
                findings.append("Declining accuracy trend over last 3 rounds")

        # Over-reliance on correction
        if latest.correction_rate > 0.5:
            findings.append(f"High correction rate: {latest.correction_rate:.1%} of tasks needed correction")

        if findings:
            logger.warning(f"Detected {len(findings)} weakness(es)")
        return findings

    def get_trend_data(self) -> List[Dict[str, float]]:
        """Return performance history suitable for plotting."""
        return [
            {"accuracy": h.accuracy, "ece": h.ece, "ccps": h.ccps}
            for h in self.history
        ]
