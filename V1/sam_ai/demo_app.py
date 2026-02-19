"""
SAM-AI  ·  Interactive Demo Interface
=======================================
Streamlit web application for demonstrating the SAM-AI cognitive reasoning
pipeline in an interactive, research-grade format.

Run with:
    streamlit run sam_ai/demo_app.py

Features:
- Natural language input via NLP parser
- Full pipeline execution (Reasoning → Meta-Eval → Uncertainty → Self-Correction)
- Structured display of all pipeline outputs
- Real-time visualisation of reasoning traces
"""

from __future__ import annotations

import os
import sys
import json
import time
from typing import Any, Dict, List, Optional

# ── Ensure the project root is on sys.path ────────────────────────────────────
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import streamlit as st

# ── SAM-AI module imports ─────────────────────────────────────────────────────
from sam_ai.reasoning_engine import ReasoningEngine
from sam_ai.meta_evaluator import MetaEvaluator
from sam_ai.uncertainty_model import UncertaintyModel
from sam_ai.self_corrector import SelfCorrector
from sam_ai.nlp_parser import NLPParser

# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SAM-AI · Cognitive Reasoning Demo",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Global font & background */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* Main container */
    .main { background: linear-gradient(135deg, #0f0c29, #302b63, #24243e); }

    /* Header */
    .hero-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 32px rgba(102,126,234,0.4);
    }
    .hero-header h1 { color: white; font-size: 2.2rem; font-weight: 700; margin: 0; }
    .hero-header p  { color: rgba(255,255,255,0.85); font-size: 1rem; margin: 0.4rem 0 0; }

    /* Section cards */
    .section-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 12px;
        padding: 1.4rem 1.6rem;
        margin-bottom: 1.2rem;
        backdrop-filter: blur(10px);
    }
    .section-title {
        font-size: 1.05rem;
        font-weight: 600;
        color: #a78bfa;
        margin-bottom: 0.8rem;
        display: flex;
        align-items: center;
        gap: 0.4rem;
    }

    /* Metric pills */
    .metric-pill {
        display: inline-block;
        background: rgba(167,139,250,0.15);
        border: 1px solid rgba(167,139,250,0.35);
        border-radius: 20px;
        padding: 0.25rem 0.75rem;
        font-size: 0.82rem;
        color: #c4b5fd;
        margin: 0.2rem;
    }

    /* Step trace rows */
    .trace-step {
        background: rgba(255,255,255,0.03);
        border-left: 3px solid #667eea;
        border-radius: 0 8px 8px 0;
        padding: 0.6rem 1rem;
        margin: 0.4rem 0;
        font-size: 0.88rem;
    }
    .trace-step .step-num { color: #818cf8; font-weight: 600; }
    .trace-step .step-rule { color: #a5f3fc; }
    .trace-step .step-result { color: #86efac; }
    .trace-step .step-conf { color: #fbbf24; }

    /* Status badges */
    .badge-pass { background:#064e3b; color:#6ee7b7; border-radius:6px; padding:2px 8px; font-size:0.8rem; }
    .badge-fail { background:#7f1d1d; color:#fca5a5; border-radius:6px; padding:2px 8px; font-size:0.8rem; }
    .badge-warn { background:#78350f; color:#fcd34d; border-radius:6px; padding:2px 8px; font-size:0.8rem; }

    /* Score bar */
    .score-bar-container { background:rgba(255,255,255,0.08); border-radius:8px; height:10px; margin:4px 0; }
    .score-bar-fill { height:10px; border-radius:8px; transition: width 0.5s ease; }

    /* Sidebar */
    [data-testid="stSidebar"] { background: rgba(15,12,41,0.95); }
    [data-testid="stSidebar"] .stMarkdown { color: #c4b5fd; }

    /* Button */
    .stButton > button {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.2s ease;
        width: 100%;
    }
    .stButton > button:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(102,126,234,0.5); }

    /* Text area */
    .stTextArea textarea {
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(167,139,250,0.3);
        border-radius: 10px;
        color: #e2e8f0;
        font-size: 0.95rem;
    }
</style>
""", unsafe_allow_html=True)


# ── Cached module instances ───────────────────────────────────────────────────
@st.cache_resource
def get_pipeline():
    """Initialise and cache all pipeline components."""
    engine    = ReasoningEngine()
    evaluator = MetaEvaluator()
    uncertainty = UncertaintyModel()
    corrector = SelfCorrector(
        reasoning_engine=engine,
        meta_evaluator=evaluator,
        uncertainty_model=uncertainty,
    )
    parser = NLPParser()
    return engine, evaluator, uncertainty, corrector, parser


# ── Helper renderers ──────────────────────────────────────────────────────────
def _score_bar(value: float, color: str = "#667eea") -> str:
    pct = max(0, min(100, value * 100))
    return (
        f'<div class="score-bar-container">'
        f'<div class="score-bar-fill" style="width:{pct:.1f}%;background:{color};"></div>'
        f'</div>'
    )


def _badge(text: str, kind: str = "pass") -> str:
    cls = {"pass": "badge-pass", "fail": "badge-fail", "warn": "badge-warn"}.get(kind, "badge-pass")
    return f'<span class="{cls}">{text}</span>'


def _flatten_trace(node: Dict, depth: int = 0) -> List[Dict]:
    """Flatten a nested trace dict into a list of step dicts with depth info."""
    rows = [{"depth": depth, **node}]
    for child in node.get("children", []):
        rows.extend(_flatten_trace(child, depth + 1))
    return rows


def render_reasoning_trace(trace_dict: Dict):
    """Section A — Reasoning Trace."""
    st.markdown('<div class="section-title">🔍 A. Reasoning Trace</div>', unsafe_allow_html=True)
    steps = _flatten_trace(trace_dict)
    for s in steps:
        indent = "&nbsp;" * (s["depth"] * 4)
        valid_icon = "✅" if s.get("valid", True) else "❌"
        conf = s.get("confidence", 0.0)
        result_str = str(s.get("result", "—"))[:120]
        desc = s.get("description", "—")
        st.markdown(
            f'<div class="trace-step">'
            f'{indent}<span class="step-num">Step {s.get("step","?")} {valid_icon}</span> &nbsp;'
            f'<span class="step-rule">{desc}</span><br>'
            f'{indent}&nbsp;&nbsp;&nbsp;'
            f'<span class="step-result">→ {result_str}</span> &nbsp;'
            f'<span class="step-conf">conf={conf:.3f}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )


def render_meta_evaluation(meta_eval):
    """Section B — Meta-Evaluation Results."""
    st.markdown('<div class="section-title">🧪 B. Meta-Evaluation Results</div>', unsafe_allow_html=True)
    d = meta_eval.to_dict()

    col1, col2 = st.columns(2)
    with col1:
        valid_badge = _badge("VALID", "pass") if d["is_valid"] else _badge("INVALID", "fail")
        st.markdown(f"**Structural Validation:** {valid_badge}", unsafe_allow_html=True)
        st.markdown(_score_bar(d["structural_score"], "#22c55e"), unsafe_allow_html=True)
        st.caption(f"Structural Score: {d['structural_score']:.4f}")

        st.markdown(f"**Consistency Check:**", unsafe_allow_html=True)
        st.markdown(_score_bar(d["consistency_score"], "#3b82f6"), unsafe_allow_html=True)
        st.caption(f"Consistency Score: {d['consistency_score']:.4f}")

    with col2:
        st.markdown(f"**Reasoning Depth Score:**", unsafe_allow_html=True)
        st.markdown(_score_bar(d["depth_score"], "#a78bfa"), unsafe_allow_html=True)
        st.caption(f"Depth Score: {d['depth_score']:.4f}")

        quality_color = "#22c55e" if d["overall_quality"] >= 0.75 else "#f59e0b" if d["overall_quality"] >= 0.5 else "#ef4444"
        st.markdown(f"**Overall Reasoning Quality:**", unsafe_allow_html=True)
        st.markdown(_score_bar(d["overall_quality"], quality_color), unsafe_allow_html=True)
        st.caption(f"Quality Score: {d['overall_quality']:.4f}")

    # Fallacy detection
    if d["issues"]:
        st.markdown("**🚨 Issues Detected:**")
        for issue in d["issues"]:
            st.markdown(f'<span class="metric-pill">⚠ {issue}</span>', unsafe_allow_html=True)
    if d["warnings"]:
        st.markdown("**⚠ Warnings:**")
        for w in d["warnings"]:
            st.markdown(f'<span class="metric-pill">⚡ {w}</span>', unsafe_allow_html=True)
    if not d["issues"] and not d["warnings"]:
        st.success("✅ No fallacies or inconsistencies detected.")


def render_uncertainty_metrics(ue):
    """Section C — Uncertainty Metrics."""
    st.markdown('<div class="section-title">📊 C. Uncertainty Metrics</div>', unsafe_allow_html=True)
    d = ue.to_dict()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Final Confidence", f"{d['calibrated_confidence']:.4f}")
        rating = d["reliability_rating"]
        color = {"HIGH": "🟢", "MODERATE": "🟡", "LOW": "🟠", "VERY_LOW": "🔴"}.get(rating, "⚪")
        st.markdown(f"**Reliability:** {color} {rating}")
    with col2:
        st.metric("Entropy Estimate", f"{d['entropy']:.4f}")
        st.caption("Shannon entropy of step confidence distribution")
    with col3:
        calib = d["calibrated_confidence"]
        raw   = d["aggregate_confidence"]
        delta = calib - raw
        st.metric("Calibration Δ", f"{delta:+.4f}")
        indicator = "Well-calibrated ✅" if abs(delta) < 0.05 else "Calibration shift ⚠"
        st.caption(indicator)

    # Step confidence sparkline
    if d["step_confidences"]:
        st.markdown("**Step Confidence Distribution:**")
        st.bar_chart({"Confidence": d["step_confidences"]}, height=120)


def render_self_correction(cr):
    """Section D — Self-Correction Loop Results."""
    st.markdown('<div class="section-title">🔄 D. Self-Correction Loop Results</div>', unsafe_allow_html=True)
    d = cr.to_dict()

    col1, col2 = st.columns(2)
    with col1:
        triggered = d["was_corrected"]
        badge = _badge("TRIGGERED", "warn") if triggered else _badge("NOT NEEDED", "pass")
        st.markdown(f"**Correction Status:** {badge}", unsafe_allow_html=True)
        st.markdown(f"**Original Answer:** `{d['original_answer']}`")
        st.markdown(f"**Final Answer:** `{d['corrected_answer']}`")

    with col2:
        improvement = d["quality_after"] - d["quality_before"]
        st.metric("Quality Before", f"{d['quality_before']:.4f}")
        st.metric("Quality After",  f"{d['quality_after']:.4f}", delta=f"{improvement:+.4f}")

    if d["correction_log"]:
        st.markdown("**Correction Rounds:**")
        for entry in d["correction_log"]:
            outcome_badge = _badge(entry.get("outcome", "?"), "pass" if entry.get("outcome") == "accepted" else "fail")
            st.markdown(
                f'<span class="metric-pill">Round {entry["round"]} | '
                f'Strategy: <b>{entry.get("strategy","?")}</b> | '
                f'Quality: {entry.get("quality", "—")} | {outcome_badge}</span>',
                unsafe_allow_html=True,
            )


# ── Sidebar ───────────────────────────────────────────────────────────────────
def render_sidebar():
    st.sidebar.markdown("## 🧠 SAM-AI Controls")
    st.sidebar.markdown("---")

    st.sidebar.markdown("### Pipeline Settings")
    enable_correction = st.sidebar.checkbox("Enable Self-Correction", value=True)
    show_raw_json     = st.sidebar.checkbox("Show Raw JSON Output", value=False)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Example Problems")
    examples = {
        "Logic (Propositional)": "If it rains, the ground is wet. It is raining. Is the ground wet?",
        "Logic (Syllogistic)":   "All mammals are animals. All dogs are mammals. Are all dogs animals?",
        "Math (Arithmetic)":     "What is 15 + 27 * 3?",
        "Math (Algebra)":        "If x + 5 = 12, what is x?",
        "Pattern (Sequence)":    "What is the next number in the sequence: 2, 4, 8, 16, __?",
        "Pattern (Analogy)":     "2 is to 4 as 3 is to what? (n→n²)",
    }
    selected = st.sidebar.selectbox("Load Example", ["— Select —"] + list(examples.keys()))

    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.info(
        "SAM-AI is a research-grade cognitive reasoning system featuring:\n"
        "- Symbolic forward-chaining\n"
        "- Meta-evaluation & fallacy detection\n"
        "- Bayesian uncertainty quantification\n"
        "- Iterative self-correction"
    )

    return enable_correction, show_raw_json, examples.get(selected, "")


# ── Main App ──────────────────────────────────────────────────────────────────
def main():
    # Header
    st.markdown("""
    <div class="hero-header">
        <h1>🧠 SAM-AI · Cognitive Reasoning System</h1>
        <p>Self-Evaluating, Self-Correcting Reasoning Intelligence — Research Demo Interface</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
    enable_correction, show_raw_json, example_text = render_sidebar()

    # Load pipeline
    engine, evaluator, uncertainty, corrector, parser = get_pipeline()

    # Input area
    st.markdown("### 💬 Enter a Reasoning Problem")
    col_input, col_btn = st.columns([4, 1])

    with col_input:
        default_text = example_text if example_text else ""
        user_input = st.text_area(
            "Describe your reasoning problem in natural language:",
            value=default_text,
            height=120,
            placeholder=(
                "Example: If it rains, the ground is wet. It is raining. Is the ground wet?\n"
                "Example: What is the next number: 2, 4, 8, 16, __?\n"
                "Example: All mammals are animals. All dogs are mammals. Are all dogs animals?"
            ),
            key="problem_input",
            label_visibility="collapsed",
        )

    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        run_clicked = st.button("🚀 Run SAM-AI\nReasoning", key="run_btn")

    # ── Run pipeline ──────────────────────────────────────────────────────────
    if run_clicked:
        if not user_input.strip():
            st.warning("⚠ Please enter a reasoning problem first.")
            return

        with st.spinner("🔄 Running SAM-AI cognitive pipeline…"):
            # 1. Parse natural language → structured task
            task = parser.parse(user_input.strip())

            # 2. Reasoning Engine
            t0 = time.time()
            result = engine.solve(task)
            t_reason = time.time() - t0

            # 3. Meta-Evaluation
            trace_dict = result.trace.to_dict()
            meta_eval  = evaluator.evaluate(trace_dict)

            # 4. Uncertainty Estimation
            ue = uncertainty.estimate(trace_dict, task.get("category", "unknown"))

            # 5. Self-Correction (if enabled)
            if enable_correction:
                cr = corrector.correct(task, result, meta_eval)
                # Recompute on corrected trace
                if cr.was_corrected and cr.final_result is not None:
                    trace_dict = cr.final_result.trace.to_dict()
                    meta_eval  = evaluator.evaluate(trace_dict)
                    ue         = uncertainty.estimate(trace_dict, task.get("category", "unknown"))
            else:
                # Build a dummy CorrectionResult for display
                from sam_ai.self_corrector import CorrectionResult
                cr = CorrectionResult()
                cr.original_answer  = result.answer
                cr.corrected_answer = result.answer
                cr.quality_before   = meta_eval.overall_quality
                cr.quality_after    = meta_eval.overall_quality
                cr.final_result     = result

        # ── Display results ───────────────────────────────────────────────────
        st.markdown("---")

        # Task info banner
        cat = task.get("category", "unknown")
        st.markdown(
            f'<div style="background:rgba(102,126,234,0.15);border-radius:10px;padding:0.8rem 1.2rem;margin-bottom:1rem;">'
            f'<b>Detected Category:</b> <span class="metric-pill">{cat}</span> &nbsp;'
            f'<b>Answer:</b> <code>{cr.corrected_answer}</code> &nbsp;'
            f'<b>Pipeline time:</b> {t_reason*1000:.1f} ms'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Four output sections in a 2×2 layout
        col_left, col_right = st.columns(2)

        with col_left:
            with st.container():
                st.markdown('<div class="section-card">', unsafe_allow_html=True)
                render_reasoning_trace(trace_dict)
                st.markdown('</div>', unsafe_allow_html=True)

            with st.container():
                st.markdown('<div class="section-card">', unsafe_allow_html=True)
                render_uncertainty_metrics(ue)
                st.markdown('</div>', unsafe_allow_html=True)

        with col_right:
            with st.container():
                st.markdown('<div class="section-card">', unsafe_allow_html=True)
                render_meta_evaluation(meta_eval)
                st.markdown('</div>', unsafe_allow_html=True)

            with st.container():
                st.markdown('<div class="section-card">', unsafe_allow_html=True)
                render_self_correction(cr)
                st.markdown('</div>', unsafe_allow_html=True)

        # Raw JSON
        if show_raw_json:
            st.markdown("---")
            st.markdown("### 🗂 Raw Pipeline Output (JSON)")
            raw = {
                "task":        task,
                "reasoning":   result.to_dict(),
                "meta_eval":   meta_eval.to_dict(),
                "uncertainty": ue.to_dict(),
                "correction":  cr.to_dict(),
            }
            st.json(raw)

    else:
        # Landing state
        st.markdown("""
        <div style="text-align:center;padding:3rem;opacity:0.6;">
            <div style="font-size:4rem;">🧠</div>
            <p style="font-size:1.1rem;color:#a78bfa;">
                Enter a reasoning problem above and click <b>Run SAM-AI Reasoning</b> to begin.
            </p>
            <p style="font-size:0.85rem;color:#6b7280;">
                Supports: Propositional Logic · Syllogistic Reasoning · Arithmetic · Algebra · Sequences · Analogies
            </p>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
