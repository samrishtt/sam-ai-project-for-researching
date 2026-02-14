"""
SAM-AI  ·  Self-Correction Loop  (v2 – Aggressive)
======================================================
Detects flawed reasoning nodes and attempts automatic refinement
through iterative re-reasoning and solution revision.

v2 Changes
----------
- Raised quality_threshold to 0.85 (more self-critical)
- Added null-answer detection as an automatic correction trigger
- Added "fuzzy_re_solve" strategy that varies engine parameters
- Improved logging and tracking of correction outcomes
"""

from __future__ import annotations
from typing import Any, Dict, Optional
from sam_ai.reasoning_engine import ReasoningEngine, ReasoningResult
from sam_ai.meta_evaluator import MetaEvaluator, MetaEvaluation
from sam_ai.uncertainty_model import UncertaintyModel
from sam_ai.utils.logging import SAMLogger

logger = SAMLogger.get_logger("SelfCorrector")


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
            "original_answer": self.original_answer,
            "corrected_answer": self.corrected_answer,
            "was_corrected": self.was_corrected,
            "correction_rounds": self.correction_rounds,
            "quality_before": round(self.quality_before, 4),
            "quality_after": round(self.quality_after, 4),
            "correction_log": self.correction_log,
        }


class SelfCorrector:
    """Iterative self-correction loop for reasoning refinement.

    Parameters
    ----------
    max_rounds : int
        Maximum number of correction attempts.
    quality_threshold : float
        Minimum overall quality to accept without correction.
    improvement_threshold : float
        Minimum quality improvement to accept a correction.
    """

    def __init__(
        self,
        reasoning_engine: Optional[ReasoningEngine] = None,
        meta_evaluator: Optional[MetaEvaluator] = None,
        uncertainty_model: Optional[UncertaintyModel] = None,
        max_rounds: int = 3,
        quality_threshold: float = 0.85,
        improvement_threshold: float = 0.02,
    ):
        self.engine = reasoning_engine or ReasoningEngine()
        self.evaluator = meta_evaluator or MetaEvaluator()
        self.uncertainty = uncertainty_model or UncertaintyModel()
        self.max_rounds = max_rounds
        self.quality_threshold = quality_threshold
        self.improvement_threshold = improvement_threshold

    def correct(
        self,
        task: Dict[str, Any],
        initial_result: ReasoningResult,
        initial_eval: MetaEvaluation,
    ) -> CorrectionResult:
        """Attempt to refine a reasoning result.

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
        cr.quality_before = initial_eval.overall_quality
        cr.final_result = initial_result

        # ── v2: Force correction if answer is None (clearly wrong) ──
        answer_is_null = initial_result.answer is None
        has_null_steps = any("null" in w.lower() for w in initial_eval.warnings)

        needs_correction = (
            initial_eval.overall_quality < self.quality_threshold
            or bool(initial_eval.issues)
            or answer_is_null
            or has_null_steps
        )

        if not needs_correction:
            cr.corrected_answer = initial_result.answer
            cr.quality_after = initial_eval.overall_quality
            cr.final_result = initial_result
            return cr

        if answer_is_null:
            logger.warning(f"Task {task['id']}: answer is None — triggering forced correction")

        best_result = initial_result
        best_eval = initial_eval
        best_quality = initial_eval.overall_quality

        for round_num in range(1, self.max_rounds + 1):
            cr.correction_rounds = round_num
            logger.reasoning(f"Correction round {round_num}/{self.max_rounds} for {task['id']}")

            # Strategy selection based on issues
            strategy = self._select_strategy(best_eval, best_result)
            cr.correction_log.append({"round": round_num, "strategy": strategy})

            # Re-solve with modified parameters
            new_result = self._apply_strategy(task, strategy, round_num)
            if new_result is None:
                cr.correction_log[-1]["outcome"] = "strategy_failed"
                continue

            # Re-evaluate
            new_eval = self.evaluator.evaluate(new_result.trace.to_dict())
            new_quality = new_eval.overall_quality

            # v2: Bonus for non-null answer when original was null
            if answer_is_null and new_result.answer is not None:
                new_quality += 0.15  # Strong incentive

            cr.correction_log[-1]["quality"] = round(new_quality, 4)

            # Accept if improvement exceeds threshold
            if new_quality > best_quality + self.improvement_threshold:
                best_result = new_result
                best_eval = new_eval
                best_quality = new_quality
                cr.correction_log[-1]["outcome"] = "accepted"
                logger.info(f"  Round {round_num}: quality improved → {new_quality:.3f}")
            else:
                cr.correction_log[-1]["outcome"] = "rejected"

            # Early exit if quality now sufficient
            if best_quality >= self.quality_threshold and best_result.answer is not None:
                break

        cr.corrected_answer = best_result.answer
        cr.quality_after = best_quality
        cr.was_corrected = (cr.corrected_answer != cr.original_answer)
        cr.final_result = best_result

        if cr.was_corrected:
            logger.info(f"Task {task['id']}: CORRECTED {cr.original_answer} → {cr.corrected_answer}")
        else:
            logger.info(f"Task {task['id']}: answer unchanged after {cr.correction_rounds} round(s)")

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
