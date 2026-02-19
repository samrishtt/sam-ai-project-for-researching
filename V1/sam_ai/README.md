# SAM-AI: Self-Evaluating, Self-Correcting Reasoning Intelligence System

> *An experimental cognitive architecture for multi-step reasoning, meta-evaluation, uncertainty quantification, and iterative self-correction.*

---

## 1. Problem Statement

Contemporary AI systems excel at pattern-matching and statistical prediction but remain fundamentally limited in their capacity for **reflective reasoning** — the ability to monitor, evaluate, and correct their own cognitive processes. This gap represents one of the most significant barriers on the path toward Artificial General Intelligence (AGI).

**SAM-AI** addresses this challenge by implementing a modular *cognitive pipeline* that generates structured reasoning traces, subjects them to meta-evaluation, quantifies uncertainty, and iteratively refines solutions through self-correction loops. The system serves as an experimental platform for studying the emergence of cognitive self-awareness in artificial agents.

---

## 2. Research Motivation

### 2.1 Meta-Cognition in AI

Human cognition is distinguished not only by the ability to reason but by the capacity to *reason about reasoning*. This meta-cognitive faculty enables:

- Error detection and correction during problem-solving
- Confidence calibration ("knowing what you know")
- Strategy selection based on self-assessed competence
- Learning from mistakes through reflective analysis

SAM-AI simulates these processes algorithmically, providing a testbed for investigating which architectural components are necessary and sufficient for meta-cognitive capability.

### 2.2 Relation to AGI Research

This work draws on several threads in AGI research:

| Research Area | SAM-AI Component |
|---|---|
| Chain-of-Thought Reasoning | Reasoning Engine |
| Self-Consistency Checking | Meta-Evaluation Engine |
| Bayesian Uncertainty | Uncertainty Model |
| Constitutional AI / Self-Improvement | Self-Correction Loop |
| Benchmarking & Evaluation | Performance Analytics |

---

## 3. System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        SAM-AI COGNITIVE PIPELINE                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   ┌───────────────┐     ┌──────────────────┐                       │
│   │  Task Loader   │────▶│  REASONING       │                       │
│   │  (Datasets)    │     │  ENGINE          │                       │
│   └───────────────┘     │  • Forward-chain  │                       │
│                          │  • Domain solvers │                       │
│                          │  • Trace builder  │                       │
│                          └────────┬─────────┘                       │
│                                   │ Reasoning Trace                  │
│                                   ▼                                  │
│                          ┌──────────────────┐                       │
│                          │  META-EVALUATION  │                       │
│                          │  ENGINE           │                       │
│                          │  • Structure check│                       │
│                          │  • Consistency    │                       │
│                          │  • Fallacy detect │                       │
│                          └────────┬─────────┘                       │
│                                   │ Quality Assessment               │
│                                   ▼                                  │
│                          ┌──────────────────┐                       │
│                          │  UNCERTAINTY      │                       │
│                          │  MODEL            │                       │
│                          │  • Bayesian conf. │                       │
│                          │  • Calibration    │                       │
│                          │  • Entropy est.   │                       │
│                          └────────┬─────────┘                       │
│                                   │ Confidence Scores                │
│                                   ▼                                  │
│                     ┌────────────────────────────┐                  │
│                     │  SELF-CORRECTION LOOP       │                  │
│                     │  • Strategy selection        │                  │
│                     │  • Re-solve & re-evaluate    │                  │
│                     │  • Accept if improved        │◀──── Feedback   │
│                     └────────────┬───────────────┘       Loop       │
│                                  │ Final Answer                      │
│                                  ▼                                   │
│                     ┌────────────────────────────┐                  │
│                     │  PERFORMANCE ANALYTICS      │                  │
│                     │  • Accuracy tracking         │                  │
│                     │  • Calibration analysis       │                  │
│                     │  • Weakness detection         │                  │
│                     │  • Trend visualisation        │                  │
│                     └────────────────────────────┘                  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 4. Module Descriptions

### 4.1 Reasoning Engine (`reasoning_engine.py`)

The core cognitive module. Implements a **symbolic forward-chainer** that:

- Parses task structure (premises, questions, constraints)
- Decomposes problems into atomic sub-goals
- Applies domain-specific inference rules (propositional logic, algebra, sequence detection, etc.)
- Produces annotated reasoning traces with per-step confidence scores

**Supported domains:** Propositional logic, syllogistic reasoning, conditional/contrapositive inference, arithmetic, algebra, number theory, word problems, sequence extrapolation, matrix patterns, analogies.

### 4.2 Meta-Evaluation Engine (`meta_evaluator.py`)

Performs *reasoning about reasoning*:

- **Structural validation** — ensures trace integrity (ordered steps, no missing fields)
- **Consistency analysis** — detects contradictory intermediate results
- **Fallacy detection** — identifies circular reasoning, non-sequiturs, and affirming the consequent
- **Quality scoring** — aggregates structural, consistency, depth, and confidence metrics

### 4.3 Uncertainty Model (`uncertainty_model.py`)

Assigns probabilistic confidence estimates:

- **Depth-decayed confidence** — deeper reasoning chains accumulate uncertainty
- **Geometric aggregation** — weak links in the chain penalise overall confidence
- **Temperature-scaled calibration** — Platt-style sigmoid rescaling
- **Shannon entropy** — measures information spread across reasoning steps

### 4.4 Self-Correction Loop (`self_corrector.py`)

Iterative refinement mechanism:

1. Evaluates whether a reasoning result meets quality thresholds
2. Selects a repair strategy (deep re-solve, depth increase, consistency repair)
3. Re-solves and re-evaluates
4. Accepts corrections only if quality demonstrably improves

### 4.5 Performance Analytics (`performance_analyzer.py`)

Longitudinal performance tracking:

- **Accuracy** and **Expected Calibration Error (ECE)** per round
- **Composite Cognitive Performance Score (CCPS)** integrating multiple metrics
- **Per-category breakdown** for targeted weakness detection
- **Trend analysis** for identifying declining performance patterns

---

## 5. Evaluation Methodology

### 5.1 Benchmark Tasks

SAM-AI includes 45 curated tasks across three domains:

| Domain | Categories | # Tasks | Difficulty Range |
|---|---|---|---|
| Logic | Propositional, Syllogistic, Conditional, Contrapositive | 15 | 1–4 |
| Mathematics | Arithmetic, Algebra, Number Theory, Word Problems | 15 | 1–3 |
| Pattern | Sequence, Matrix, Analogy | 15 | 1–3 |

### 5.2 Metrics

| Metric | Description |
|---|---|
| Accuracy | Exact-match correctness |
| ECE | Expected Calibration Error (10-bin) |
| CCPS | Composite Cognitive Performance Score |
| Chain Consistency | Weighted validity of reasoning traces |
| Correction Rate | Fraction of tasks requiring self-correction |

### 5.3 Experimental Protocol

1. Load benchmark tasks
2. Generate reasoning traces via the Reasoning Engine
3. Validate traces via the Meta-Evaluation Engine
4. Estimate confidence via the Uncertainty Model
5. Apply Self-Correction Loop
6. Record performance via the Performance Analyzer
7. Generate cognitive reasoning report and diagnostic plots

---

## 6. Experimental Results Format

Each run produces:

- **Console report** — per-task reasoning traces with annotations
- **`performance_trends.png`** — accuracy, ECE, and CCPS over rounds
- **`calibration_diagram.png`** — reliability diagram showing confidence vs. accuracy
- **`performance_history.jsonl`** — machine-readable record of all rounds

---

## 7. Installation & Usage

### Prerequisites

- Python 3.9+

### Setup

```bash
cd sam_ai
pip install -r requirements.txt
```

### Run

```bash
# Full evaluation (all domains, 1 round)
python -m sam_ai.main

# Multiple rounds
python -m sam_ai.main --rounds 3

# Specific domains only
python -m sam_ai.main --domains logic math

# Without self-correction
python -m sam_ai.main --no-correction

# Custom output directory
python -m sam_ai.main --output results/exp_01
```

---

## 8. Future Research Directions

1. **Neural–Symbolic Hybrid Reasoning** — Integrate neural language models for premise understanding while retaining symbolic verification.

2. **Learned Meta-Evaluation** — Train the meta-evaluator on human-annotated reasoning quality judgements.

3. **Adversarial Reasoning Tasks** — Generate tasks designed to exploit known fallacy patterns and test robustness.

4. **Hierarchical Self-Improvement** — Implement a second-order meta-loop that optimises the correction strategy selector itself.

5. **Multi-Agent Deliberation** — Extend the architecture to support multiple reasoning agents that debate and converge.

6. **Temporal Reasoning** — Add support for tasks involving temporal ordering, causality, and counterfactual inference.

7. **Explainability Interfaces** — Generate natural-language explanations of reasoning traces for human inspection.

8. **Transfer Learning Across Domains** — Investigate whether reasoning patterns learned in one domain generalise to others.

---

## 9. Project Structure

```
sam_ai/
├── main.py                         # CLI entry point
├── reasoning_engine.py             # Structured reasoning generation
├── meta_evaluator.py               # Reasoning trace validation
├── uncertainty_model.py            # Confidence estimation
├── self_corrector.py               # Iterative refinement loop
├── performance_analyzer.py         # Longitudinal analytics
├── datasets/
│   ├── logic_tasks.py              # Propositional & syllogistic tasks
│   ├── math_reasoning.py           # Arithmetic, algebra, word problems
│   └── pattern_tasks.py            # Sequence & matrix patterns
├── utils/
│   ├── scoring.py                  # Evaluation metrics
│   ├── logging.py                  # Structured logging facility
│   └── visualization.py           # Plots & report generation
├── experiments/
│   └── evaluation_runner.py        # Experiment orchestrator
├── requirements.txt
└── README.md
```

---

## 10. Citation

If you use SAM-AI in your research, please cite:

```bibtex
@software{sam_ai_2026,
  title     = {SAM-AI: Self-Evaluating, Self-Correcting Reasoning Intelligence System},
  author    = {SAM-AI Research Team},
  year      = {2026},
  url       = {https://github.com/sam-ai-research/sam-ai},
  note      = {Experimental cognitive architecture for meta-reasoning research}
}
```

---

## 11. Performance Milestones

| Date       | Accuracy | ECE   | Milestones Achieved |
|------------|----------|-------|---------------------|
| 2026-02-14 | **100.0%** | 0.05  | Full reasoning pipeline refinement; resolved ambiguity in analogy tasks and implicit multiplication in polynomial evaluation. |
| 2026-02-18 | —        | —     | Research-grade upgrade: interactive demo, NLP parser, adversarial dataset, adaptive learning, 4-mode baseline comparison, extended visualizations. |

---

## 12. Research-Grade Upgrade (v3)

This section documents the seven-stage upgrade that transforms SAM-AI into a fully demonstrable, research-grade cognitive reasoning system.

### Stage 1 — Interactive Demo Interface (`sam_ai/demo_app.py`)

A Streamlit web application providing a live demonstration of the full SAM-AI pipeline.

**Features:**
- Natural language input via the NLP Parser
- Four output panels: Reasoning Trace · Meta-Evaluation · Uncertainty Metrics · Self-Correction
- Sidebar controls: enable/disable correction, load example problems, toggle raw JSON output
- Dark glassmorphism UI with score bars and confidence sparklines

**Run:**
```bash
streamlit run sam_ai/demo_app.py
```

---

### Stage 2 — Baseline Experiment Comparison (`sam_ai/experiments/evaluation_runner.py`)

The `EvaluationRunner` now supports four ablation modes for systematic component analysis:

| Mode | Components Active | Purpose |
|------|-------------------|---------|
| 1 | Reasoning Engine only | Baseline without evaluation |
| 2 | + Meta-Evaluation | Adds quality scoring |
| 3 | + Uncertainty Model | Adds calibrated confidence |
| 4 | Full Pipeline (+ Self-Correction) | Complete SAM-AI system |

**Run all modes with comparison table:**
```python
from sam_ai.experiments.evaluation_runner import BaselineComparisonRunner
runner = BaselineComparisonRunner(output_dir="output", domains=["logic", "math"])
results = runner.run_all_modes()
```

**CLI:**
```bash
python sam_ai/main.py --mode compare --domains logic math pattern
```

---

### Stage 3 — Performance Visualization (`sam_ai/utils/visualization.py`)

Three new publication-quality charts (dark theme, saved to `output/`):

| Chart | File | Description |
|-------|------|-------------|
| Accuracy vs Iteration | `accuracy_vs_iteration.png` | Accuracy trend with linear regression overlay |
| Error Reduction | `error_reduction.png` | Before/after correction error rates by category |
| Module Contribution | `module_contribution.png` | Accuracy/ECE/CCPS across all 4 evaluation modes |

---

### Stage 4 — NLP Preprocessing Module (`sam_ai/nlp_parser.py`)

Rule-based natural language → structured task converter. No external APIs or ML models required.

**Supported categories:** propositional · syllogistic · conditional · contrapositive · arithmetic · algebra · number_theory · word_problem · sequence · matrix · analogy

**Usage:**
```python
from sam_ai.nlp_parser import NLPParser
parser = NLPParser()
task = parser.parse("If it rains, the ground is wet. It is raining. Is the ground wet?")
# → {"category": "propositional", "premises": [...], "question": "...", ...}
```

---

### Stage 5 — Adaptive Learning Mechanism (`sam_ai/self_corrector.py`)

The `SelfCorrector` (v3) now learns from past correction patterns across sessions.

**How it works:**
1. After each task, records outcome to `output/learning_history.json`
2. Tracks per-category failure rates and per-strategy acceptance rates
3. Adjusts confidence thresholds: `effective_threshold = quality_threshold × confidence_weight`
4. Prefers historically successful strategies for each category

**Access learning statistics:**
```python
from sam_ai.self_corrector import SelfCorrector
sc = SelfCorrector()
print(sc.get_learning_statistics())
```

---

### Stage 6 — Adversarial Dataset (`sam_ai/datasets/adversarial_tasks.py`)

19 curated adversarial tasks across three categories:

| Type | Count | Examples |
|------|-------|---------|
| `fallacy_trap` | 7 | Affirming the consequent, circular reasoning, hasty generalisation |
| `contradictory` | 5 | Direct contradiction, implicit contradiction chains |
| `ambiguous` | 7 | Scope ambiguity, referential ambiguity, Liar's paradox |

**Usage:**
```python
from sam_ai.datasets.adversarial_tasks import get_adversarial_tasks, get_adversarial_tasks_by_type
all_tasks = get_adversarial_tasks()                          # all 19
fallacies = get_adversarial_tasks_by_type("fallacy_trap")   # 7 tasks
```

Include in evaluation:
```python
runner = EvaluationRunner(domains=["logic", "math", "pattern", "adversarial"])
```

---

### Stage 7 — Updated Documentation

This README has been updated to document all new modules, APIs, and usage patterns introduced in the v3 research-grade upgrade.

---

## License

This project is released for academic research purposes.

---

*SAM-AI — Toward machines that think about their own thinking.*
