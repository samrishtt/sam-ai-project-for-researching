"""
SAM-AI  ·  Full Pipeline Test & Results Collection
=====================================================
Runs all evaluation modes and collects data for the research paper.
"""
import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "sam_ai"))
sys.path.insert(0, os.path.dirname(__file__))

print("=" * 70)
print("  SAM-AI  ·  FULL PIPELINE TEST")
print("=" * 70)

# ── Test 1: NLP Parser ──────────────────────────────────────────
print("\n[TEST 1] NLP Parser")
from sam_ai.nlp_parser import NLPParser
parser = NLPParser()

test_inputs = [
    ("If it rains, the ground is wet. It is raining. Is the ground wet?", "propositional"),
    ("All mammals are animals. All dogs are mammals. Are all dogs animals?", "syllogistic"),
    ("What is 15 + 27?", "arithmetic"),
    ("Solve for x: x + 5 = 12", "algebra"),
    ("What is the next number in the sequence: 2, 4, 8, 16, __?", "sequence"),
    ("Is 17 a prime number?", "number_theory"),
]

nlp_results = []
for text, expected_cat in test_inputs:
    task = parser.parse(text)
    match = task["category"] == expected_cat
    status = "PASS" if match else "FAIL"
    print(f"  [{status}] '{text[:50]}...' → {task['category']} (expected: {expected_cat})")
    nlp_results.append({"input": text, "detected": task["category"], "expected": expected_cat, "match": match})

nlp_accuracy = sum(1 for r in nlp_results if r["match"]) / len(nlp_results)
print(f"  NLP Parser Accuracy: {nlp_accuracy:.0%}")

# ── Test 2: Adversarial Dataset ─────────────────────────────────
print("\n[TEST 2] Adversarial Dataset")
from sam_ai.datasets.adversarial_tasks import get_adversarial_tasks, get_adversarial_summary
tasks = get_adversarial_tasks()
summary = get_adversarial_summary()
print(f"  Total adversarial tasks: {len(tasks)}")
print(f"  By type: {summary}")

# ── Test 3: Single-task pipeline run ────────────────────────────
print("\n[TEST 3] Single-task pipeline run")
from sam_ai.reasoning_engine import ReasoningEngine
from sam_ai.meta_evaluator import MetaEvaluator
from sam_ai.uncertainty_model import UncertaintyModel
from sam_ai.self_corrector import SelfCorrector

engine = ReasoningEngine()
evaluator = MetaEvaluator()
uncertainty = UncertaintyModel()
corrector = SelfCorrector(reasoning_engine=engine, meta_evaluator=evaluator, uncertainty_model=uncertainty)

test_task = {
    "id": "TEST-001",
    "category": "propositional",
    "premises": ["If it rains, the ground is wet.", "It is raining."],
    "question": "Is the ground wet?",
    "answer": True,
    "difficulty": 1,
}

t0 = time.time()
result = engine.solve(test_task)
t_reason = time.time() - t0

trace_dict = result.trace.to_dict()
meta_eval = evaluator.evaluate(trace_dict)
ue = uncertainty.estimate(trace_dict, "propositional")
cr = corrector.correct(test_task, result, meta_eval)

print(f"  Answer: {cr.corrected_answer} (expected: {test_task['answer']})")
print(f"  Correct: {cr.corrected_answer == test_task['answer']}")
print(f"  Quality: {cr.quality_after:.4f}")
print(f"  Confidence: {ue.calibrated_confidence:.4f}")
print(f"  Reliability: {ue.reliability_rating}")
print(f"  Time: {t_reason*1000:.1f}ms")

# ── Test 4: Full evaluation run (all domains) ──────────────────
print("\n[TEST 4] Full Evaluation Run (Mode 4 — Full Pipeline)")
from sam_ai.experiments.evaluation_runner import EvaluationRunner, MODE_FULL_PIPELINE

runner = EvaluationRunner(
    output_dir="output",
    enable_correction=True,
    domains=["logic", "math", "pattern"],
    mode=MODE_FULL_PIPELINE,
)
full_summary = runner.run(round_id=1)
print(f"\n  Accuracy: {full_summary.get('accuracy', 0):.1%}")
print(f"  ECE: {full_summary.get('ece', 0):.4f}")
print(f"  CCPS: {full_summary.get('ccps', 0):.4f}")
print(f"  Correction Rate: {full_summary.get('correction_rate', 0):.1%}")

# ── Test 5: Adversarial evaluation ──────────────────────────────
print("\n[TEST 5] Adversarial Evaluation")
adv_runner = EvaluationRunner(
    output_dir="output",
    enable_correction=True,
    domains=["adversarial"],
    mode=MODE_FULL_PIPELINE,
)
adv_summary = adv_runner.run(round_id=2)
print(f"\n  Adversarial Accuracy: {adv_summary.get('accuracy', 0):.1%}")
print(f"  Adversarial ECE: {adv_summary.get('ece', 0):.4f}")

# ── Test 6: Multi-mode comparison ───────────────────────────────
print("\n[TEST 6] Multi-Mode Baseline Comparison")
from sam_ai.experiments.evaluation_runner import (
    EvaluationRunner, MODE_REASONING_ONLY, MODE_REASONING_META,
    MODE_REASONING_META_UNC, MODE_FULL_PIPELINE, MODE_NAMES
)

mode_results = {}
for mode in [MODE_REASONING_ONLY, MODE_REASONING_META, MODE_REASONING_META_UNC, MODE_FULL_PIPELINE]:
    mr = EvaluationRunner(
        output_dir="output",
        enable_correction=(mode == MODE_FULL_PIPELINE),
        domains=["logic", "math", "pattern"],
        mode=mode,
    )
    s = mr.run(round_id=mode)
    mode_results[mode] = s
    print(f"  {MODE_NAMES[mode]}: acc={s.get('accuracy',0):.3f}, ece={s.get('ece',0):.4f}, ccps={s.get('ccps',0):.4f}")

# ── Save all results to JSON ────────────────────────────────────
results_data = {
    "nlp_parser": {"accuracy": nlp_accuracy, "results": nlp_results},
    "adversarial_dataset": {"total_tasks": len(tasks), "summary": summary},
    "single_task": {
        "answer": str(cr.corrected_answer),
        "correct": cr.corrected_answer == test_task["answer"],
        "quality": cr.quality_after,
        "confidence": ue.calibrated_confidence,
        "reliability": ue.reliability_rating,
    },
    "full_pipeline": full_summary,
    "adversarial_eval": adv_summary,
    "mode_comparison": {str(k): v for k, v in mode_results.items()},
}

os.makedirs("output", exist_ok=True)
with open("output/test_results.json", "w") as f:
    json.dump(results_data, f, indent=2, default=str)

print("\n" + "=" * 70)
print("  ALL TESTS COMPLETE — Results saved to output/test_results.json")
print("=" * 70)
