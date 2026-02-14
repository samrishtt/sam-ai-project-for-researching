"""
SAM-AI  ·  Evaluation Runner  (v2)
=====================================
Orchestrates full experimental runs: loads tasks, invokes the
cognitive pipeline, collects results, and produces reports.

v2 Changes
----------
- Uses corrected result's trace + confidence when correction occurs
- Passes enable_correction flag to control correction behaviour
"""

from __future__ import annotations
import time
from typing import Any, Dict, List, Optional

from sam_ai.reasoning_engine import ReasoningEngine
from sam_ai.meta_evaluator import MetaEvaluator
from sam_ai.uncertainty_model import UncertaintyModel
from sam_ai.self_corrector import SelfCorrector
from sam_ai.performance_analyzer import PerformanceAnalyzer
from sam_ai.datasets.logic_tasks import get_logic_tasks
from sam_ai.datasets.math_reasoning import get_math_tasks
from sam_ai.datasets.pattern_tasks import get_pattern_tasks
from sam_ai.utils.visualization import (
    print_cognitive_report,
    print_summary_table,
    plot_performance_trends,
    plot_calibration_diagram,
)
from sam_ai.utils.logging import SAMLogger

logger = SAMLogger.get_logger("EvaluationRunner")


class EvaluationRunner:
    """Full experimental evaluation orchestrator.

    Parameters
    ----------
    output_dir : str
        Directory for generated plots and logs.
    enable_correction : bool
        Whether to run the self-correction loop.
    domains : list of str
        Which task domains to include (``logic``, ``math``, ``pattern``).
    """

    def __init__(
        self,
        output_dir: str = "output",
        enable_correction: bool = True,
        domains: Optional[List[str]] = None,
    ):
        self.output_dir = output_dir
        self.enable_correction = enable_correction
        self.domains = domains or ["logic", "math", "pattern"]

        # Pipeline components
        self.engine = ReasoningEngine()
        self.evaluator = MetaEvaluator()
        self.uncertainty = UncertaintyModel()
        self.corrector = SelfCorrector(
            reasoning_engine=self.engine,
            meta_evaluator=self.evaluator,
            uncertainty_model=self.uncertainty,
        )
        self.analyzer = PerformanceAnalyzer(
            history_path=f"{output_dir}/performance_history.jsonl"
        )

    def load_tasks(self) -> List[Dict[str, Any]]:
        """Load tasks from selected domains."""
        tasks = []
        if "logic" in self.domains:
            tasks.extend(get_logic_tasks())
        if "math" in self.domains:
            tasks.extend(get_math_tasks())
        if "pattern" in self.domains:
            tasks.extend(get_pattern_tasks())
        logger.info(f"Loaded {len(tasks)} tasks from domains: {self.domains}")
        return tasks

    def run(self, round_id: int = 1) -> Dict[str, Any]:
        """Execute a full evaluation round.

        Returns
        -------
        dict
            Summary of the evaluation round.
        """
        tasks = self.load_tasks()
        task_results: List[Dict[str, Any]] = []
        all_confs: List[float] = []
        all_correct: List[bool] = []

        start_time = time.time()
        logger.info(f"═══ Starting evaluation round {round_id} ═══")

        for task in tasks:
            # 1. Reasoning
            result = self.engine.solve(task)

            # 2. Meta-evaluation
            trace_dict = result.trace.to_dict()
            meta_eval = self.evaluator.evaluate(trace_dict)

            # 3. Uncertainty estimation
            ue = self.uncertainty.estimate(trace_dict, task.get("category", ""))

            # 4. Self-correction (if enabled)
            final_answer = result.answer
            was_corrected = False
            correction_info = {}
            if self.enable_correction:
                cr = self.corrector.correct(task, result, meta_eval)
                final_answer = cr.corrected_answer
                was_corrected = cr.was_corrected
                correction_info = cr.to_dict()

                # v2: If corrected, recompute uncertainty on the new trace
                if was_corrected and cr.final_result is not None:
                    trace_dict = cr.final_result.trace.to_dict()
                    meta_eval = self.evaluator.evaluate(trace_dict)
                    ue = self.uncertainty.estimate(trace_dict, task.get("category", ""))

            # 5. Collect results
            correct = final_answer == task["answer"]
            confidence = ue.calibrated_confidence

            entry = {
                "task_id": task["id"],
                "category": task.get("category", "unknown"),
                "question": task.get("question", ""),
                "prediction": final_answer,
                "ground_truth": task["answer"],
                "correct": correct,
                "confidence": confidence,
                "step_validities": meta_eval.step_validities,
                "step_confidences": meta_eval.step_confidences,
                "meta_quality": meta_eval.overall_quality,
                "reliability": ue.reliability_rating,
                "was_corrected": was_corrected,
                "trace": trace_dict,
                "meta_notes": meta_eval.issues + meta_eval.warnings,
                "correction": correction_info,
            }
            task_results.append(entry)
            all_confs.append(confidence)
            all_correct.append(correct)

        elapsed = time.time() - start_time

        # 6. Record performance
        record = self.analyzer.record_batch(round_id, task_results)
        weaknesses = self.analyzer.detect_weaknesses()

        # 7. Generate report
        summary = record.to_dict()
        summary["elapsed_seconds"] = round(elapsed, 2)
        summary["weaknesses"] = weaknesses

        print_cognitive_report(task_results, summary)

        # 8. Generate plots
        try:
            trend_data = self.analyzer.get_trend_data()
            if len(trend_data) >= 1:
                plot_performance_trends(
                    trend_data,
                    output_path=f"{self.output_dir}/performance_trends.png",
                )
            plot_calibration_diagram(
                all_confs, all_correct,
                output_path=f"{self.output_dir}/calibration_diagram.png",
            )
        except Exception as e:
            logger.warning(f"Plot generation failed: {e}")

        logger.info(f"═══ Round {round_id} complete: {elapsed:.1f}s ═══")
        return summary
