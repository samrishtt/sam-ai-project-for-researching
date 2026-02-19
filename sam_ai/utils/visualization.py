"""
SAM-AI  ·  Visualization Toolkit  (v2 – Extended)
===================================================
Generates publication-quality figures for:

- Reasoning-trace tree diagrams (text-based)
- Performance trend plots (accuracy, ECE over time)
- Confidence calibration diagrams
- Cognitive report summaries (console & optional PNG)
- [NEW] Accuracy vs Iteration graph
- [NEW] Error reduction after correction loops
- [NEW] Module contribution comparison chart
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence

try:
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    import numpy as np
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

from tabulate import tabulate


# ─── Text-based Reasoning Trace Tree ────────────────────────────
def render_reasoning_tree(trace: Dict[str, Any], indent: int = 0) -> str:
    """Recursively render a reasoning trace as an indented tree string.

    Expected trace schema::

        {
            "step": int,
            "description": str,
            "result": Any,
            "confidence": float,
            "valid": bool,
            "children": [<sub-trace>, ...]   # optional
        }
    """
    prefix = "│   " * indent
    marker = "✓" if trace.get("valid", True) else "✗"
    conf = trace.get("confidence", 0.0)
    line = (
        f"{prefix}├─ Step {trace['step']}: {trace['description']}  "
        f"[{marker}  conf={conf:.2f}]"
    )
    result_line = f"{prefix}│   → result: {trace.get('result', '—')}"
    lines = [line, result_line]
    for child in trace.get("children", []):
        lines.append(render_reasoning_tree(child, indent + 1))
    return "\n".join(lines)


# ─── Console Summary Table ──────────────────────────────────────
def print_summary_table(metrics: Dict[str, float], title: str = "Performance Summary"):
    """Pretty-print a metrics dictionary as a table."""
    rows = [[k, f"{v:.4f}" if isinstance(v, float) else str(v)] for k, v in metrics.items()]
    header = ["Metric", "Value"]
    print(f"\n{'─'*50}")
    print(f"  {title}")
    print(f"{'─'*50}")
    print(tabulate(rows, headers=header, tablefmt="rounded_outline"))
    print()


# ─── Performance Trend Plot ─────────────────────────────────────
def plot_performance_trends(
    history: List[Dict[str, float]],
    output_path: str = "performance_trends.png",
):
    """Plot accuracy, ECE, and CCPS over evaluation rounds.

    Parameters
    ----------
    history : list of dict
        Each dict must contain keys ``accuracy``, ``ece``, ``ccps``.
    output_path : str
        File path for saved figure.
    """
    if not HAS_MPL:
        print("[visualization] matplotlib not available – skipping plot.")
        return

    rounds = list(range(1, len(history) + 1))
    acc  = [h["accuracy"] for h in history]
    ece  = [h["ece"] for h in history]
    ccps = [h["ccps"] for h in history]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    fig.suptitle("SAM-AI  ·  Cognitive Performance Trends", fontsize=14, fontweight="bold")

    # Accuracy
    axes[0].plot(rounds, acc, "o-", color="#2ecc71", linewidth=2)
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Round")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].grid(alpha=0.3)

    # ECE
    axes[1].plot(rounds, ece, "s-", color="#e74c3c", linewidth=2)
    axes[1].set_title("Expected Calibration Error")
    axes[1].set_xlabel("Round")
    axes[1].set_ylabel("ECE")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].grid(alpha=0.3)

    # CCPS
    axes[2].plot(rounds, ccps, "D-", color="#3498db", linewidth=2)
    axes[2].set_title("Cognitive Performance Score")
    axes[2].set_xlabel("Round")
    axes[2].set_ylabel("CCPS")
    axes[2].set_ylim(-0.05, 1.05)
    axes[2].grid(alpha=0.3)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[visualization] Saved trend plot → {output_path}")


# ─── Calibration Diagram ────────────────────────────────────────
def plot_calibration_diagram(
    confidences: Sequence[float],
    correctness: Sequence[bool],
    n_bins: int = 10,
    output_path: str = "calibration_diagram.png",
):
    """Reliability diagram (calibration plot) with gap bars."""
    if not HAS_MPL:
        print("[visualization] matplotlib not available – skipping plot.")
        return

    bin_boundaries = [i / n_bins for i in range(n_bins + 1)]
    bin_accs, bin_confs, bin_sizes = [], [], []

    for b in range(n_bins):
        lo, hi = bin_boundaries[b], bin_boundaries[b + 1]
        indices = [
            i for i, c in enumerate(confidences)
            if (lo <= c < hi) or (b == n_bins - 1 and c == hi)
        ]
        if not indices:
            bin_accs.append(0)
            bin_confs.append((lo + hi) / 2)
            bin_sizes.append(0)
            continue
        bin_accs.append(sum(1 for i in indices if correctness[i]) / len(indices))
        bin_confs.append(sum(confidences[i] for i in indices) / len(indices))
        bin_sizes.append(len(indices))

    fig, ax = plt.subplots(figsize=(6, 5))
    width = 1 / n_bins
    positions = [(bin_boundaries[b] + bin_boundaries[b + 1]) / 2 for b in range(n_bins)]

    ax.bar(positions, bin_accs, width=width * 0.9, color="#3498db", alpha=0.7, label="Accuracy")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")
    ax.set_xlabel("Mean predicted confidence")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("SAM-AI  ·  Confidence Calibration Diagram")
    ax.legend(loc="upper left")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[visualization] Saved calibration diagram → {output_path}")


# ─── [NEW] Accuracy vs Iteration ────────────────────────────────
def plot_accuracy_vs_iteration(
    history: List[Dict[str, float]],
    output_path: str = "accuracy_vs_iteration.png",
):
    """
    Plot accuracy across evaluation iterations/rounds with trend line.

    Parameters
    ----------
    history : list of dict
        Each dict must contain key ``accuracy``.
    output_path : str
        File path for saved figure.
    """
    if not HAS_MPL:
        print("[visualization] matplotlib not available – skipping plot.")
        return

    rounds = list(range(1, len(history) + 1))
    acc    = [h.get("accuracy", 0.0) for h in history]

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#1e293b")

    # Main accuracy line
    ax.plot(rounds, acc, "o-", color="#22d3ee", linewidth=2.5,
            markersize=8, markerfacecolor="#0ea5e9", label="Accuracy")

    # Trend line (linear regression if enough points)
    if len(rounds) >= 3 and HAS_MPL:
        try:
            z = np.polyfit(rounds, acc, 1)
            p = np.poly1d(z)
            ax.plot(rounds, p(rounds), "--", color="#f59e0b", linewidth=1.5,
                    alpha=0.8, label=f"Trend (slope={z[0]:+.3f})")
        except Exception:
            pass

    # Shaded region
    ax.fill_between(rounds, acc, alpha=0.15, color="#22d3ee")

    ax.set_title("SAM-AI  ·  Accuracy vs Iteration", color="white", fontsize=13, fontweight="bold")
    ax.set_xlabel("Evaluation Round", color="#94a3b8")
    ax.set_ylabel("Accuracy", color="#94a3b8")
    ax.set_ylim(-0.05, 1.05)
    ax.tick_params(colors="#94a3b8")
    ax.spines["bottom"].set_color("#334155")
    ax.spines["left"].set_color("#334155")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.15, color="#334155")
    ax.legend(facecolor="#1e293b", labelcolor="white", framealpha=0.8)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[visualization] Saved accuracy vs iteration → {output_path}")


# ─── [NEW] Error Reduction After Correction Loops ───────────────
def plot_error_reduction(
    task_results: List[Dict[str, Any]],
    output_path: str = "error_reduction.png",
):
    """
    Bar chart showing error rates before and after self-correction,
    broken down by task category.

    Parameters
    ----------
    task_results : list of dict
        Each dict must contain: ``category``, ``correct``, ``was_corrected``,
        ``correction`` (with ``quality_before`` and ``quality_after``).
    output_path : str
        File path for saved figure.
    """
    if not HAS_MPL:
        print("[visualization] matplotlib not available – skipping plot.")
        return

    # Aggregate by category
    from collections import defaultdict
    cat_data: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {"total": 0, "errors_before": 0, "errors_after": 0, "corrected": 0}
    )

    for tr in task_results:
        cat = tr.get("category", "unknown")
        cat_data[cat]["total"] += 1

        # Error before correction = not correct before any correction
        # We approximate: if was_corrected, the original was wrong
        if tr.get("was_corrected"):
            cat_data[cat]["errors_before"] += 1
            cat_data[cat]["corrected"] += 1
        elif not tr.get("correct"):
            cat_data[cat]["errors_before"] += 1

        # Error after = still not correct
        if not tr.get("correct"):
            cat_data[cat]["errors_after"] += 1

    if not cat_data:
        return

    categories = sorted(cat_data.keys())
    n = len(categories)
    x = list(range(n))

    errors_before = [cat_data[c]["errors_before"] / max(1, cat_data[c]["total"]) for c in categories]
    errors_after  = [cat_data[c]["errors_after"]  / max(1, cat_data[c]["total"]) for c in categories]

    fig, ax = plt.subplots(figsize=(max(8, n * 1.2), 5))
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#1e293b")

    bar_w = 0.35
    bars1 = ax.bar([xi - bar_w/2 for xi in x], errors_before, bar_w,
                   label="Error Rate Before Correction", color="#ef4444", alpha=0.85)
    bars2 = ax.bar([xi + bar_w/2 for xi in x], errors_after, bar_w,
                   label="Error Rate After Correction",  color="#22c55e", alpha=0.85)

    # Value labels
    for bar in bars1:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.01, f"{h:.0%}",
                    ha="center", va="bottom", color="white", fontsize=8)
    for bar in bars2:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.01, f"{h:.0%}",
                    ha="center", va="bottom", color="white", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=30, ha="right", color="#94a3b8", fontsize=9)
    ax.set_title("SAM-AI  ·  Error Reduction After Self-Correction", color="white",
                 fontsize=13, fontweight="bold")
    ax.set_ylabel("Error Rate", color="#94a3b8")
    ax.set_ylim(0, 1.15)
    ax.tick_params(colors="#94a3b8")
    ax.spines["bottom"].set_color("#334155")
    ax.spines["left"].set_color("#334155")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.15, color="#334155")
    ax.legend(facecolor="#1e293b", labelcolor="white", framealpha=0.8)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[visualization] Saved error reduction chart → {output_path}")


# ─── [NEW] Module Contribution Comparison Chart ──────────────────
def plot_module_contribution(
    mode_results: Dict[int, Dict[str, Any]],
    output_path: str = "module_contribution.png",
):
    """
    Grouped bar chart comparing key metrics across the 4 evaluation modes,
    illustrating the contribution of each pipeline module.

    Parameters
    ----------
    mode_results : dict
        Mapping of mode_id (1–4) → summary dict from EvaluationRunner.run().
    output_path : str
        File path for saved figure.
    """
    if not HAS_MPL:
        print("[visualization] matplotlib not available – skipping plot.")
        return

    from sam_ai.experiments.evaluation_runner import MODE_NAMES

    modes    = sorted(mode_results.keys())
    labels   = [MODE_NAMES.get(m, f"Mode {m}").replace("Mode ", "M").split(":")[0] for m in modes]
    metrics  = ["accuracy", "ccps", "ece"]
    m_labels = ["Accuracy", "CCPS", "ECE (lower=better)"]
    colors   = ["#22d3ee", "#a78bfa", "#f87171"]

    n_modes   = len(modes)
    n_metrics = len(metrics)
    x = list(range(n_modes))
    bar_w = 0.22

    fig, ax = plt.subplots(figsize=(max(10, n_modes * 2.5), 5))
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#1e293b")

    for mi, (metric, label, color) in enumerate(zip(metrics, m_labels, colors)):
        values = [mode_results[m].get(metric, 0.0) for m in modes]
        offsets = [xi + (mi - n_metrics/2 + 0.5) * bar_w for xi in x]
        bars = ax.bar(offsets, values, bar_w, label=label, color=color, alpha=0.85)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", color="white", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, color="#94a3b8", fontsize=9)
    ax.set_title("SAM-AI  ·  Module Contribution Comparison", color="white",
                 fontsize=13, fontweight="bold")
    ax.set_ylabel("Score", color="#94a3b8")
    ax.set_ylim(0, 1.2)
    ax.tick_params(colors="#94a3b8")
    ax.spines["bottom"].set_color("#334155")
    ax.spines["left"].set_color("#334155")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.15, color="#334155")
    ax.legend(facecolor="#1e293b", labelcolor="white", framealpha=0.8, fontsize=9)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[visualization] Saved module contribution chart → {output_path}")


# ─── Full Cognitive Report (Console) ────────────────────────────
def print_cognitive_report(
    task_results: List[Dict[str, Any]],
    summary: Dict[str, float],
):
    """Print a detailed per-task reasoning report followed by aggregate
    metrics.  Designed for terminal readability.
    """
    print("\n" + "═" * 72)
    print("  SAM-AI  ·  COGNITIVE REASONING REPORT")
    print("═" * 72)

    for i, tr in enumerate(task_results, 1):
        status    = "CORRECT ✓" if tr.get("correct") else "INCORRECT ✗"
        corrected = "  (self-corrected)" if tr.get("was_corrected") else ""
        print(f"\n── Task {i}: {tr.get('task_id', 'N/A')} ── {status}{corrected}")
        print(f"   Question : {tr.get('question', '—')}")
        print(f"   Answer   : {tr.get('prediction')}  (expected: {tr.get('ground_truth')})")
        print(f"   Confidence: {tr.get('confidence', 0.0):.3f}")

        # Reasoning trace
        trace = tr.get("trace")
        if trace:
            print("   Reasoning Trace:")
            print(render_reasoning_tree(trace))

        # Meta-evaluation notes
        notes = tr.get("meta_notes", [])
        if notes:
            print("   Meta-Evaluation Notes:")
            for note in notes:
                print(f"     ⚠  {note}")

    print_summary_table(summary)
    print("═" * 72 + "\n")
