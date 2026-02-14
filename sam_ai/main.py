"""
SAM-AI  ·  Main Entry Point
==============================
Command-line interface for the Self-Evaluating, Self-Correcting
Reasoning Intelligence System.

Usage
-----
    python -m sam_ai.main                     # run all domains, 1 round
    python -m sam_ai.main --rounds 3          # 3 evaluation rounds
    python -m sam_ai.main --domains logic math # only logic + math
    python -m sam_ai.main --no-correction     # disable self-correction
"""

from __future__ import annotations

import argparse
import os
import sys
import time

from sam_ai.experiments.evaluation_runner import EvaluationRunner
from sam_ai.utils.logging import SAMLogger
from sam_ai.utils.visualization import print_summary_table


def print_banner():
    banner = r"""
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║   ███████╗ █████╗ ███╗   ███╗       █████╗ ██╗               ║
    ║   ██╔════╝██╔══██╗████╗ ████║      ██╔══██╗██║               ║
    ║   ███████╗███████║██╔████╔██║█████╗███████║██║               ║
    ║   ╚════██║██╔══██║██║╚██╔╝██║╚════╝██╔══██║██║               ║
    ║   ███████║██║  ██║██║ ╚═╝ ██║      ██║  ██║██║               ║
    ║   ╚══════╝╚═╝  ╚═╝╚═╝     ╚═╝      ╚═╝  ╚═╝╚═╝               ║
    ║                                                              ║
    ║   Self-Evaluating, Self-Correcting Reasoning Intelligence    ║
    ║   ─────────────────────────────────────────────────────────   ║
    ║   Cognitive Architecture for Meta-Reasoning Research         ║
    ║   v0.1.0                                                     ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def main():
    parser = argparse.ArgumentParser(
        prog="sam-ai",
        description="SAM-AI: Self-Evaluating, Self-Correcting Reasoning Intelligence",
    )
    parser.add_argument(
        "--rounds", type=int, default=1,
        help="Number of evaluation rounds to run (default: 1)",
    )
    parser.add_argument(
        "--domains", nargs="+", default=["logic", "math", "pattern"],
        choices=["logic", "math", "pattern"],
        help="Task domains to evaluate (default: all)",
    )
    parser.add_argument(
        "--no-correction", action="store_true",
        help="Disable the self-correction loop",
    )
    parser.add_argument(
        "--output", type=str, default="output",
        help="Output directory for plots and logs (default: output)",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    print_banner()

    # Setup
    os.makedirs(args.output, exist_ok=True)
    logger = SAMLogger.get_logger("Main")

    logger.info("Initialising SAM-AI cognitive pipeline…")
    logger.info(f"  Domains   : {args.domains}")
    logger.info(f"  Rounds    : {args.rounds}")
    logger.info(f"  Correction: {'ENABLED' if not args.no_correction else 'DISABLED'}")
    logger.info(f"  Output    : {args.output}")

    runner = EvaluationRunner(
        output_dir=args.output,
        enable_correction=not args.no_correction,
        domains=args.domains,
    )

    # Run evaluation rounds
    all_summaries = []
    total_start = time.time()

    for r in range(1, args.rounds + 1):
        summary = runner.run(round_id=r)
        all_summaries.append(summary)

    total_elapsed = time.time() - total_start

    # Final summary
    print("\n" + "═" * 72)
    print("  EXPERIMENT COMPLETE")
    print("═" * 72)
    print(f"  Total rounds  : {args.rounds}")
    print(f"  Total time    : {total_elapsed:.1f}s")

    if all_summaries:
        final = all_summaries[-1]
        print(f"  Final accuracy: {final.get('accuracy', 0):.1%}")
        print(f"  Final CCPS    : {final.get('ccps', 0):.4f}")
        print(f"  Final ECE     : {final.get('ece', 0):.4f}")

        weaknesses = final.get("weaknesses", [])
        if weaknesses:
            print("\n  ⚠  Detected Weaknesses:")
            for w in weaknesses:
                print(f"     • {w}")

    print("═" * 72 + "\n")
    logger.info("SAM-AI session complete.")


if __name__ == "__main__":
    main()
