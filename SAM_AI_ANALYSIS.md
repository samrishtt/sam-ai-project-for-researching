# SAM-AI Code Analysis: Cognitive Pipeline Architecture

This document provides a technical deep-dive into the internal mechanics of SAM-AI, analyzing how it implements multi-step symbolic reasoning, meta-evaluation, and self-correction.

## 1. Overview of the Cognitive Loop

SAM-AI is designed as a **closed-loop cognitive system**. Unlike standard LLMs which perform single-pass probabilistic inference, SAM-AI treats reasoning as a structured, verifiable, and iterative process.

The pipeline follows these distinct stages:
1. **Perception**: NLP Parser converts natural language to structured tasks.
2. **Cognition**: Reasoning Engine generates a symbolic deduction trace.
3. **Metacognition (Verification)**: Meta-Evaluator checks the trace for fallacies and structural integrity.
4. **Metacognition (Calibration)**: Uncertainty Model estimates the probability of correctness.
5. **Action (Correction)**: Self-Corrector modifies the reasoning path if quality is sub-par.
6. **Learning**: The system stores failure patterns to adjust future correction thresholds.

---

## 2. The Reasoning Engine (`reasoning_engine.py`)

The engine is a **Symbolic Forward-Chainer**. It does not "guess" tokens; it applies formal rules to premises.

### A. Logic Domain
- **Propositional Logic**: Implements *Modus Ponens* (if P → Q and P, then Q) and *Modus Tollens* (if P → Q and ¬Q, then ¬P). It uses **fuzzy semantic matching** to link premises that are worded differently but mean the same thing (e.g., "the sun shines" vs "the sun is shining").
- **Syllogistic Reasoning**: Uses set-theoretic closure. It builds a directed graph of relationships (All A are B → A ⊆ B) and computes the **transitive closure** to derive non-obvious conclusions.
- **Disjunctive Inference**: Handles "Either P or Q" scenarios by resolving the negation of one branch to prove the other.

### B. Math Domain
- Uses a recursive parser to evaluate expressions.
- Handles algebra by isolating variables and applying standard arithmetic precedence.
- **Word Problem Solver**: Extracts numeric constants and operators from text via regex and rule-based templates.

### C. Pattern Domain
- **Sequence Extrapolation**: Identifies constant differences (arithmetic), constant ratios (geometric), and power-based patterns (n²).
- **Analogy Solver**: Uses a multi-strategy approach. It first checks for explicit hints (e.g., "n → n³"), then falls back to detecting letter-position sums or numeric ratios.

---

## 3. Meta-Evaluation Engine (`meta_evaluator.py`)

This module performs **Verification through Structural Analysis**. It doesn't look at the answer; it looks at the *path* taken to get there.

- **Fallacy Detection**: It scans for "Affirming the Consequent" (if P → Q and Q, then P — which is invalid) and "Circular Reasoning" (where a conclusion is used as its own premise).
- **Consistency Check**: It ensures that Step 5 does not contradict facts established in Step 2.
- **Quality Score ($Q$)**: A weighted average of structural integrity, logical consistency, and step-wise confidence.

---

## 4. Uncertainty Model (`uncertainty_model.py`)

SAM-AI uses a **Bayesian-inspired Confidence Calibration**.

- **Depth Decay**: As a reasoning chain grows longer, the cumulative uncertainty increases. $C_{total} = C_{base} \times (1 - \epsilon)^d$, where $d$ is depth.
- **Weak Link Penalty**: The aggregate confidence is constrained by the lowest-confidence step in the chain ($min(C_i)$).
- **Entropy Estimation**: Measures the "spread" of confidence. High entropy indicates the system is wavering between multiple interpretations.

---

## 5. Self-Correction Loop (`self_corrector.py`)

The **v3 Adaptive Self-Corrector** is the most advanced component.

1. **Trigger**: If $Q < 0.85$ or an explicit fallacy is detected, the loop starts.
2. **Strategy Selection**: It looks at the `learning_history.json`. If "Logic" tasks have frequently failed with "Simple Resolve", it switches to "Deep Chaining".
3. **Execution**: It re-invokes the Reasoning Engine with stricter constraints or alternate heuristics.
4. **Validation**: The *new* trace is re-evaluated. If the new $Q_{new} > Q_{old}$, the correction is accepted.

---

## 6. Conclusion of Analysis

SAM-AI represents a shift from "Probabilistic Prediction" to **"Symbolic Verification"**. By externalizing the reasoning process into a trace and subjecting it to a distinct "critic" module, the system achieves a level of reliability and self-awareness that is fundamentally different from standard autoregressive models.
