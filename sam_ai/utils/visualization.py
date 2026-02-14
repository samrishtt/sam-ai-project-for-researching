"""
SAM-AI  ·  Visualization Toolkit
==================================
Generates publication-quality figures for:

- Reasoning-trace tree diagrams (text-based)
- Performance trend plots (accuracy, ECE over time)
- Confidence calibration diagrams
- Cognitive report summaries (console & optional PNG)
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence

try:
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
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
    acc = [h["accuracy"] for h in history]
    ece = [h["ece"] for h in history]
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
        status = "CORRECT ✓" if tr.get("correct") else "INCORRECT ✗"
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
