"""
SAM-AI  ·  Evaluation Runner  (v3 – Baseline Comparison Modes)
================================================================
Orchestrates full experimental runs: loads tasks, invokes the
cognitive pipeline, collects results, and produces reports.

v2 Changes
----------
- Uses corrected result's trace + confidence when correction occurs
- Passes enable_correction flag to control correction behaviour

v3 Changes (Baseline Comparison)
----------------------------------
- Supports 4 evaluation modes for ablation study:
    Mode 1: Reasoning Engine Only
    Mode 2: Reasoning + Meta Evaluation
    Mode 3: Reasoning + Meta Evaluation + Uncertainty Model
    Mode 4: Full SAM-AI Pipeline (all components)
- Records Accuracy, ECE, Correction Rate, Consistency Score per mode
- Auto-generates a comparison table after all modes complete
- Integrates adversarial task dataset
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
from sam_ai.datasets.adversarial_tasks import get_adversarial_tasks
from sam_ai.utils.visualization import (
    print_cognitive_report,
    print_summary_table,
    plot_performance_trends,
    plot_calibration_diagram,
    plot_accuracy_vs_iteration,
    plot_error_reduction,
    plot_module_contribution,
)
from sam_ai.utils.logging import SAMLogger

logger = SAMLogger.get_logger("EvaluationRunner")


# ══════════════════════════════════════════════════════════════════════════════
#  Mode constants
# ══════════════════════════════════════════════════════════════════════════════

MODE_REASONING_ONLY        = 1
MODE_REASONING_META        = 2
MODE_REASONING_META_UNC    = 3
MODE_FULL_PIPELINE         = 4

MODE_NAMES = {
    MODE_REASONING_ONLY:     "Mode 1: Reasoning Engine Only",
    MODE_REASONING_META:     "Mode 2: Reasoning + Meta Evaluation",
    MODE_REASONING_META_UNC: "Mode 3: Reasoning + Meta Eval + Uncertainty",
    MODE_FULL_PIPELINE:      "Mode 4: Full SAM-AI Pipeline",
}


# ══════════════════════════════════════════════════════════════════════════════
#  EvaluationRunner
# ══════════════════════════════════════════════════════════════════════════════

class EvaluationRunner:
    """
    Full experimental evaluation orchestrator.

    Parameters
    ----------
    output_dir : str
        Directory for generated plots and logs.
    enable_correction : bool
        Whether to run the self-correction loop.
    domains : list of str
        Which task domains to include (``logic``, ``math``, ``pattern``,
        ``adversarial``).
    mode : int
        Evaluation mode (1–4). See MODE_* constants.
    """

    def __init__(
        self,
        output_dir: str = "output",
        enable_correction: bool = True,
        domains: Optional[List[str]] = None,
        mode: int = MODE_FULL_PIPELINE,
    ):
        self.output_dir = output_dir
        self.enable_correction = enable_correction
        self.domains = domains or ["logic", "math", "pattern"]
        self.mode = mode

        # Pipeline components
        self.engine    = ReasoningEngine()
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
        if "adversarial" in self.domains:
            tasks.extend(get_adversarial_tasks())
        logger.info(f"Loaded {len(tasks)} tasks from domains: {self.domains}")
        return tasks

    def run(self, round_id: int = 1) -> Dict[str, Any]:
        """
        Execute a full evaluation round.

        Returns
        -------
        dict
            Summary of the evaluation round.
        """
        tasks = self.load_tasks()
        task_results: List[Dict[str, Any]] = []
        all_confs:    List[float] = []
        all_correct:  List[bool]  = []

        start_time = time.time()
        mode_name  = MODE_NAMES.get(self.mode, f"Mode {self.mode}")
        logger.info(f"═══ Starting evaluation round {round_id} [{mode_name}] ═══")

        for task in tasks:
            entry = self._process_task(task)
            task_results.append(entry)
            all_confs.append(entry["confidence"])
            all_correct.append(entry["correct"])

        elapsed = time.time() - start_time

        # 6. Record performance
        record     = self.analyzer.record_batch(round_id, task_results)
        weaknesses = self.analyzer.detect_weaknesses()

        # 7. Generate report
        summary = record.to_dict()
        summary["elapsed_seconds"] = round(elapsed, 2)
        summary["weaknesses"]      = weaknesses
        summary["mode"]            = self.mode
        summary["mode_name"]       = mode_name

        print_cognitive_report(task_results, summary)

        # 8. Generate plots
        try:
            trend_data = self.analyzer.get_trend_data()
            if len(trend_data) >= 1:
                plot_performance_trends(
                    trend_data,
                    output_path=f"{self.output_dir}/performance_trends.png",
                )
                plot_accuracy_vs_iteration(
                    trend_data,
                    output_path=f"{self.output_dir}/accuracy_vs_iteration.png",
                )
            plot_calibration_diagram(
                all_confs, all_correct,
                output_path=f"{self.output_dir}/calibration_diagram.png",
            )
            plot_error_reduction(
                task_results,
                output_path=f"{self.output_dir}/error_reduction.png",
            )
        except Exception as e:
            logger.warning(f"Plot generation failed: {e}")

        logger.info(f"═══ Round {round_id} complete: {elapsed:.1f}s ═══")
        return summary

    def _process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single task through the pipeline according to the current mode."""
        # ── Step 1: Reasoning (always) ────────────────────────────────────────
        result = self.engine.solve(task)

        # ── Step 2: Meta-evaluation ───────────────────────────────────────────
        trace_dict = result.trace.to_dict()
        if self.mode >= MODE_REASONING_META:
            meta_eval = self.evaluator.evaluate(trace_dict)
        else:
            # Dummy meta-eval for mode 1
            from sam_ai.meta_evaluator import MetaEvaluation
            meta_eval = MetaEvaluation()
            meta_eval.overall_quality = result.overall_confidence

        # ── Step 3: Uncertainty estimation ───────────────────────────────────
        if self.mode >= MODE_REASONING_META_UNC:
            ue = self.uncertainty.estimate(trace_dict, task.get("category", ""))
            confidence = ue.calibrated_confidence
        else:
            # Use raw engine confidence for modes 1–2
            from sam_ai.uncertainty_model import UncertaintyEstimate
            ue = UncertaintyEstimate()
            ue.calibrated_confidence = result.overall_confidence
            ue.aggregate_confidence  = result.overall_confidence
            ue.reliability_rating    = "UNKNOWN"
            ue.entropy               = 0.0
            confidence = result.overall_confidence

        # ── Step 4: Self-correction ───────────────────────────────────────────
        final_answer    = result.answer
        was_corrected   = False
        correction_info = {}

        if self.mode == MODE_FULL_PIPELINE and self.enable_correction:
            cr = self.corrector.correct(task, result, meta_eval)
            final_answer    = cr.corrected_answer
            was_corrected   = cr.was_corrected
            correction_info = cr.to_dict()

            # Recompute on corrected trace
            if was_corrected and cr.final_result is not None:
                trace_dict = cr.final_result.trace.to_dict()
                meta_eval  = self.evaluator.evaluate(trace_dict)
                ue         = self.uncertainty.estimate(trace_dict, task.get("category", ""))
                confidence = ue.calibrated_confidence

        # ── Step 5: Collect results ───────────────────────────────────────────
        correct = (final_answer == task["answer"])

        return {
            "task_id":        task["id"],
            "category":       task.get("category", "unknown"),
            "question":       task.get("question", ""),
            "prediction":     final_answer,
            "ground_truth":   task["answer"],
            "correct":        correct,
            "confidence":     confidence,
            "step_validities":  meta_eval.step_validities,
            "step_confidences": meta_eval.step_confidences,
            "meta_quality":   meta_eval.overall_quality,
            "consistency_score": meta_eval.consistency_score,
            "reliability":    ue.reliability_rating,
            "was_corrected":  was_corrected,
            "trace":          trace_dict,
            "meta_notes":     meta_eval.issues + meta_eval.warnings,
            "correction":     correction_info,
            "mode":           self.mode,
        }


# ══════════════════════════════════════════════════════════════════════════════
#  Baseline Comparison Runner
# ══════════════════════════════════════════════════════════════════════════════

class BaselineComparisonRunner:
    """
    Runs all 4 evaluation modes and produces a comparison table.

    Usage
    -----
    >>> runner = BaselineComparisonRunner(output_dir="output")
    >>> results = runner.run_all_modes()
    >>> runner.print_comparison_table(results)
    """

    def __init__(
        self,
        output_dir: str = "output",
        domains: Optional[List[str]] = None,
    ):
        self.output_dir = output_dir
        self.domains    = domains or ["logic", "math", "pattern"]

    def run_all_modes(self) -> Dict[int, Dict[str, Any]]:
        """
        Execute all 4 evaluation modes and return their summaries.

        Returns
        -------
        dict
            Mapping of mode_id → summary dict.
        """
        results: Dict[int, Dict[str, Any]] = {}

        for mode in [
            MODE_REASONING_ONLY,
            MODE_REASONING_META,
            MODE_REASONING_META_UNC,
            MODE_FULL_PIPELINE,
        ]:
            mode_name = MODE_NAMES[mode]
            print(f"\n{'═'*60}")
            print(f"  Running {mode_name}")
            print(f"{'═'*60}")

            runner = EvaluationRunner(
                output_dir=self.output_dir,
                enable_correction=(mode == MODE_FULL_PIPELINE),
                domains=self.domains,
                mode=mode,
            )
            summary = runner.run(round_id=mode)
            results[mode] = summary

        self.print_comparison_table(results)

        # Generate module contribution chart
        try:
            plot_module_contribution(
                results,
                output_path=f"{self.output_dir}/module_contribution.png",
            )
        except Exception as e:
            logger.warning(f"Module contribution plot failed: {e}")

        return results

    def print_comparison_table(self, results: Dict[int, Dict[str, Any]]):
        """Print a formatted comparison table across all modes."""
        try:
            from tabulate import tabulate
        except ImportError:
            print("[comparison] tabulate not available — skipping table.")
            return

        rows = []
        headers = ["Mode", "Accuracy", "ECE", "Correction Rate", "Consistency"]

        for mode, summary in sorted(results.items()):
            mode_name = MODE_NAMES.get(mode, f"Mode {mode}")
            accuracy  = summary.get("accuracy", 0.0)
            ece       = summary.get("ece", 0.0)
            # Correction rate: fraction of tasks that were corrected
            corr_rate = summary.get("correction_rate", 0.0)
            # Consistency: average meta_quality across tasks
            consistency = summary.get("avg_meta_quality", summary.get("ccps", 0.0))

            rows.append([
                mode_name,
                f"{accuracy:.1%}",
                f"{ece:.4f}",
                f"{corr_rate:.1%}",
                f"{consistency:.4f}",
            ])

        print("\n" + "═" * 80)
        print("  SAM-AI  ·  BASELINE COMPARISON TABLE")
        print("═" * 80)
        print(tabulate(rows, headers=headers, tablefmt="rounded_outline"))
        print("═" * 80 + "\n")
