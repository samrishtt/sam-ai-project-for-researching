import sys, os, logging
logging.disable(logging.CRITICAL)
os.environ["SAM_AI_LOG_LEVEL"] = "CRITICAL"

from sam_ai.reasoning_engine import ReasoningEngine
from sam_ai.datasets.logic_tasks import get_logic_tasks
from sam_ai.datasets.math_reasoning import get_math_tasks
from sam_ai.datasets.pattern_tasks import get_pattern_tasks

engine = ReasoningEngine()
tasks = get_logic_tasks() + get_math_tasks() + get_pattern_tasks()
fails = 0
for t in tasks:
    r = engine.solve(t)
    ok = r.answer == t["answer"]
    if not ok:
        fails += 1
        print(f"FAIL {t['id']:12s} cat={t.get('category','?'):15s} pred={r.answer!r:8s} expected={t['answer']!r}")
        if t.get("premises"):
            for p in t["premises"]:
                print(f"  premise: {p}")
        print(f"  question: {t['question']}")
        print()

print(f"\n{fails} failing out of {len(tasks)} total ({100*(len(tasks)-fails)/len(tasks):.1f}% accuracy)")
