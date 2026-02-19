"""
SAM-AI  ·  Self-Correction Loop  (v3 – Adaptive Learning)
============================================================
Detects flawed reasoning nodes and attempts automatic refinement
through iterative re-reasoning and solution revision.

v2 Changes
----------
- Raised quality_threshold to 0.85 (more self-critical)
- Added null-answer detection as an automatic correction trigger
- Added "fuzzy_re_solve" strategy that varies engine parameters
- Improved logging and tracking of correction outcomes

v3 Changes (Adaptive Learning)
-------------------------------
- Logs reasoning failures to output/learning_history.json
- Tracks correction patterns across sessions
- Adjusts rule confidence weights based on past error frequency
- Exposes learning statistics for analysis
"""

from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional

from sam_ai.reasoning_engine import ReasoningEngine, ReasoningResult
from sam_ai.meta_evaluator import MetaEvaluator, MetaEvaluation
from sam_ai.uncertainty_model import UncertaintyModel
from sam_ai.utils.logging import SAMLogger

logger = SAMLogger.get_logger("SelfCorrector")

# Default path for learning history persistence
_DEFAULT_HISTORY_PATH = os.path.join("output", "learning_history.json")


# ══════════════════════════════════════════════════════════════════════════════
#  Data Structures
# ══════════════════════════════════════════════════════════════════════════════

class CorrectionResult:
    """Encapsulates the outcome of a self-correction attempt."""
    def __init__(self):
        self.original_answer: Any = None
        self.corrected_answer: Any = None
        self.was_corrected: bool = False
        self.correction_rounds: int = 0
        self.quality_before: float = 0.0
        self.quality_after: float = 0.0
        self.correction_log: list = []
        self.final_result: Optional[ReasoningResult] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_answer":  self.original_answer,
            "corrected_answer": self.corrected_answer,
            "was_corrected":    self.was_corrected,
            "correction_rounds": self.correction_rounds,
            "quality_before":   round(self.quality_before, 4),
            "quality_after":    round(self.quality_after, 4),
            "correction_log":   self.correction_log,
        }


# ══════════════════════════════════════════════════════════════════════════════
#  Learning History Manager
# ══════════════════════════════════════════════════════════════════════════════

class LearningHistoryManager:
    """
    Persists and analyses correction patterns across sessions.

    Stored structure (output/learning_history.json):
    {
        "total_tasks":         int,
        "total_corrections":   int,
        "category_failures":   {category: count},
        "strategy_outcomes":   {strategy: {"accepted": int, "rejected": int}},
        "confidence_adjustments": {category: float},  # multiplicative weight
        "failure_log":         [{"timestamp", "task_id", "category", "strategy", "outcome"}]
    }
    """

    def __init__(self, history_path: str = _DEFAULT_HISTORY_PATH):
        self.history_path = history_path
        self._data = self._load()

    # ── Persistence ───────────────────────────────────────────────────────────
    def _load(self) -> Dict[str, Any]:
        """Load history from disk, or initialise fresh."""
        if os.path.exists(self.history_path):
            try:
                with open(self.history_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                logger.warning("Could not load learning history — starting fresh.")
        return {
            "total_tasks":            0,
            "total_corrections":      0,
            "category_failures":      {},
            "strategy_outcomes":      {},
            "confidence_adjustments": {},
            "failure_log":            [],
        }

    def save(self):
        """Persist current history to disk."""
        os.makedirs(os.path.dirname(self.history_path) or ".", exist_ok=True)
        try:
            with open(self.history_path, "w", encoding="utf-8") as f:
                json.dump(self._data, f, indent=2, default=str)
        except IOError as e:
            logger.error(f"Could not save learning history: {e}")

    # ── Recording ─────────────────────────────────────────────────────────────
    def record_task(self, task_id: str, category: str, was_corrected: bool,
                    correction_log: List[Dict]):
        """Record the outcome of processing one task."""
        self._data["total_tasks"] += 1
        if was_corrected:
            self._data["total_corrections"] += 1
            # Track category-level failure frequency
            cat_failures = self._data["category_failures"]
            cat_failures[category] = cat_failures.get(category, 0) + 1

            # Track strategy outcomes
            for entry in correction_log:
                strategy = entry.get("strategy", "unknown")
                outcome  = entry.get("outcome", "unknown")
                strat_data = self._data["strategy_outcomes"].setdefault(
                    strategy, {"accepted": 0, "rejected": 0, "failed": 0}
                )
                if outcome in strat_data:
                    strat_data[outcome] += 1
                else:
                    strat_data["failed"] += 1

            # Append to failure log (keep last 500 entries)
            self._data["failure_log"].append({
                "timestamp":      time.strftime("%Y-%m-%dT%H:%M:%S"),
                "task_id":        task_id,
                "category":       category,
                "correction_log": correction_log,
            })
            if len(self._data["failure_log"]) > 500:
                self._data["failure_log"] = self._data["failure_log"][-500:]

        # Recompute confidence adjustments after recording
        self._update_confidence_adjustments()
        self.save()

    # ── Adaptive Confidence Weights ───────────────────────────────────────────
    def _update_confidence_adjustments(self):
        """
        Compute per-category confidence weight adjustments.

        Formula:
            failure_rate = category_failures[cat] / total_tasks
            adjustment   = max(0.70, 1.0 - failure_rate * 2.0)

        A category with a 25% failure rate gets weight 0.50 (more sceptical).
        A category with 0% failure rate keeps weight 1.0.
        """
        total = max(1, self._data["total_tasks"])
        adjustments = {}
        for cat, failures in self._data["category_failures"].items():
            failure_rate = failures / total
            adjustment   = max(0.70, 1.0 - failure_rate * 2.0)
            adjustments[cat] = round(adjustment, 4)
        self._data["confidence_adjustments"] = adjustments

    def get_confidence_weight(self, category: str) -> float:
        """Return the learned confidence weight for a category (default 1.0)."""
        return self._data["confidence_adjustments"].get(category, 1.0)

    def get_best_strategy(self, category: str) -> Optional[str]:
        """
        Return the strategy with the highest acceptance rate for this category.
        Falls back to None if insufficient data.
        """
        outcomes = self._data["strategy_outcomes"]
        if not outcomes:
            return None
        best_strategy = None
        best_rate = -1.0
        for strategy, counts in outcomes.items():
            total_uses = counts.get("accepted", 0) + counts.get("rejected", 0)
            if total_uses < 2:
                continue
            rate = counts.get("accepted", 0) / total_uses
            if rate > best_rate:
                best_rate = rate
                best_strategy = strategy
        return best_strategy

    def get_statistics(self) -> Dict[str, Any]:
        """Return a summary of learning statistics."""
        total = max(1, self._data["total_tasks"])
        return {
            "total_tasks":            self._data["total_tasks"],
            "total_corrections":      self._data["total_corrections"],
            "overall_correction_rate": round(self._data["total_corrections"] / total, 4),
            "category_failure_rates": {
                cat: round(cnt / total, 4)
                for cat, cnt in self._data["category_failures"].items()
            },
            "confidence_adjustments": self._data["confidence_adjustments"],
            "strategy_outcomes":      self._data["strategy_outcomes"],
        }


# ══════════════════════════════════════════════════════════════════════════════
#  SelfCorrector
# ══════════════════════════════════════════════════════════════════════════════

class SelfCorrector:
    """
    Iterative self-correction loop for reasoning refinement.

    Parameters
    ----------
    max_rounds : int
        Maximum number of correction attempts.
    quality_threshold : float
        Minimum overall quality to accept without correction.
    improvement_threshold : float
        Minimum quality improvement to accept a correction.
    learning_history_path : str
        Path to the JSON file for persisting learning history.
    """

    def __init__(
        self,
        reasoning_engine: Optional[ReasoningEngine] = None,
        meta_evaluator: Optional[MetaEvaluator] = None,
        uncertainty_model: Optional[UncertaintyModel] = None,
        max_rounds: int = 3,
        quality_threshold: float = 0.85,
        improvement_threshold: float = 0.02,
        learning_history_path: str = _DEFAULT_HISTORY_PATH,
    ):
        self.engine    = reasoning_engine or ReasoningEngine()
        self.evaluator = meta_evaluator or MetaEvaluator()
        self.uncertainty = uncertainty_model or UncertaintyModel()
        self.max_rounds = max_rounds
        self.quality_threshold = quality_threshold
        self.improvement_threshold = improvement_threshold

        # v3: Adaptive learning manager
        self.learning = LearningHistoryManager(learning_history_path)

    def correct(
        self,
        task: Dict[str, Any],
        initial_result: ReasoningResult,
        initial_eval: MetaEvaluation,
    ) -> CorrectionResult:
        """
        Attempt to refine a reasoning result.

        Parameters
        ----------
        task : dict
            The original task specification.
        initial_result : ReasoningResult
            The first-pass reasoning output.
        initial_eval : MetaEvaluation
            Meta-evaluation of the first pass.

        Returns
        -------
        CorrectionResult
        """
        cr = CorrectionResult()
        cr.original_answer = initial_result.answer
        cr.quality_before  = initial_eval.overall_quality
        cr.final_result    = initial_result

        category = task.get("category", "unknown")

        # v3: Apply learned confidence weight to quality threshold
        conf_weight = self.learning.get_confidence_weight(category)
        effective_threshold = self.quality_threshold * conf_weight

        # ── v2: Force correction if answer is None (clearly wrong) ──
        answer_is_null = initial_result.answer is None
        has_null_steps = any("null" in w.lower() for w in initial_eval.warnings)

        needs_correction = (
            initial_eval.overall_quality < effective_threshold
            or bool(initial_eval.issues)
            or answer_is_null
            or has_null_steps
        )

        if not needs_correction:
            cr.corrected_answer = initial_result.answer
            cr.quality_after    = initial_eval.overall_quality
            cr.final_result     = initial_result
            # Record successful (no-correction) task
            self.learning.record_task(task["id"], category, False, [])
            return cr

        if answer_is_null:
            logger.warning(f"Task {task['id']}: answer is None — triggering forced correction")

        best_result  = initial_result
        best_eval    = initial_eval
        best_quality = initial_eval.overall_quality

        for round_num in range(1, self.max_rounds + 1):
            cr.correction_rounds = round_num
            logger.reasoning(f"Correction round {round_num}/{self.max_rounds} for {task['id']}")

            # v3: Prefer learned best strategy if available
            learned_strategy = self.learning.get_best_strategy(category)
            strategy = learned_strategy or self._select_strategy(best_eval, best_result)
            cr.correction_log.append({"round": round_num, "strategy": strategy})

            # Re-solve with modified parameters
            new_result = self._apply_strategy(task, strategy, round_num)
            if new_result is None:
                cr.correction_log[-1]["outcome"] = "strategy_failed"
                continue

            # Re-evaluate
            new_eval    = self.evaluator.evaluate(new_result.trace.to_dict())
            new_quality = new_eval.overall_quality

            # v2: Bonus for non-null answer when original was null
            if answer_is_null and new_result.answer is not None:
                new_quality += 0.15  # Strong incentive

            cr.correction_log[-1]["quality"] = round(new_quality, 4)

            # Accept if improvement exceeds threshold
            if new_quality > best_quality + self.improvement_threshold:
                best_result  = new_result
                best_eval    = new_eval
                best_quality = new_quality
                cr.correction_log[-1]["outcome"] = "accepted"
                logger.info(f"  Round {round_num}: quality improved → {new_quality:.3f}")
            else:
                cr.correction_log[-1]["outcome"] = "rejected"

            # Early exit if quality now sufficient
            if best_quality >= effective_threshold and best_result.answer is not None:
                break

        cr.corrected_answer = best_result.answer
        cr.quality_after    = best_quality
        cr.was_corrected    = (cr.corrected_answer != cr.original_answer)
        cr.final_result     = best_result

        if cr.was_corrected:
            logger.info(f"Task {task['id']}: CORRECTED {cr.original_answer} → {cr.corrected_answer}")
        else:
            logger.info(f"Task {task['id']}: answer unchanged after {cr.correction_rounds} round(s)")

        # v3: Record outcome for adaptive learning
        self.learning.record_task(task["id"], category, cr.was_corrected, cr.correction_log)

        return cr

    def _select_strategy(self, evaluation: MetaEvaluation, result: ReasoningResult) -> str:
        """Choose a correction strategy based on identified issues."""
        # v2: null answer always triggers a deep re-solve
        if result.answer is None:
            return "deep_re_solve"
        if not evaluation.is_valid:
            return "deep_re_solve"
        if evaluation.consistency_score < 0.7:
            return "consistency_repair"
        if evaluation.depth_score < 0.5:
            return "depth_increase"
        if evaluation.confidence_variance > 0.1:
            return "confidence_normalise"
        return "re_solve"

    def _apply_strategy(
        self, task: Dict, strategy: str, round_num: int = 1,
    ) -> Optional[ReasoningResult]:
        """Execute the chosen correction strategy."""
        try:
            if strategy == "deep_re_solve":
                # v2: Progressively increase depth & lower confidence threshold
                engine = ReasoningEngine(
                    default_confidence=max(0.60, self.engine.default_confidence - 0.05 * round_num),
                    max_chain_depth=self.engine.max_chain_depth + 5 * round_num,
                )
                return engine.solve(task)

            elif strategy == "depth_increase":
                engine = ReasoningEngine(
                    default_confidence=self.engine.default_confidence,
                    max_chain_depth=self.engine.max_chain_depth + 10,
                )
                return engine.solve(task)

            elif strategy in ("consistency_repair", "confidence_normalise", "re_solve"):
                # v2: Create a fresh engine instance to avoid caching effects
                engine = ReasoningEngine(
                    default_confidence=self.engine.default_confidence,
                    max_chain_depth=self.engine.max_chain_depth,
                )
                return engine.solve(task)

            return None
        except Exception as e:
            logger.error(f"Strategy '{strategy}' failed: {e}")
            return None

    def get_learning_statistics(self) -> Dict[str, Any]:
        """Return current adaptive learning statistics."""
        return self.learning.get_statistics()
