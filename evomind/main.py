"""CLI entrypoint.

Usage:
    python -m evomind.main --data path/to/dataset --task "Find useful patterns" \\
        --iterations 5 --threshold 0.8 --verbose --save --report
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from evomind.config import load_config
from evomind.graph import run_evomind
from evomind.telemetry import RunTracker, generate_run_report, setup_logging

RESULTS_DIR = Path(__file__).resolve().parent.parent / "experiments" / "results"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="EvoMind: an agent that evolves its own analysis strategy.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m evomind.main --data data/file.csv --task "Find useful patterns"
  python -m evomind.main --data data/sales.parquet --task "Detect anomalies" --iterations 10
  python -m evomind.main --data data/records.json --task "Cluster users" --verbose --save --report
        """,
    )
    parser.add_argument("--data", required=True, help="Path to a dataset (CSV, Parquet, JSON, JSONL, Excel, TSV).")
    parser.add_argument("--task", required=True, help="Natural-language description of the task.")
    parser.add_argument("--iterations", type=int, default=5, help="Max evolve iterations (default: 5).")
    parser.add_argument("--threshold", type=float, default=0.8, help="Score threshold to stop early (default: 0.8).")
    parser.add_argument("--db", default=None, help="Path to the SQLite strategy memory (default: data/evomind.db).")
    parser.add_argument("--save", action="store_true", help="Save the final result JSON under experiments/results/.")
    parser.add_argument("--report", action="store_true", help="Generate a markdown run report.")
    parser.add_argument("--verbose", action="store_true", help="Enable detailed structured logging.")
    parser.add_argument("--provider", default=None, help="LLM provider override (groq, huggingface, anthropic).")
    parser.add_argument("--model", default=None, help="LLM model override.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = parse_args(argv)

    # Setup
    cfg = load_config(
        provider=args.provider,
        model=args.model,
        max_iterations=args.iterations,
        score_threshold=args.threshold,
        verbose=args.verbose,
        save_results=args.save,
        generate_report=args.report,
    )
    setup_logging(verbose=cfg.verbose)

    data_path = Path(args.data)
    if not data_path.exists() and not args.data.startswith(("http://", "https://", "sqlite:///")):
        print(f"error: dataset not found at {args.data}", file=sys.stderr)
        return 1

    memory = None
    if args.db:
        from evomind.memory import StrategyMemory
        memory = StrategyMemory(args.db)

    # LLM client
    llm = None
    if args.provider or args.model:
        from evomind.llm import get_llm_client
        llm = get_llm_client(provider=args.provider, model=args.model)

    # Telemetry
    tracker = RunTracker()

    print(f"🧬 EvoMind — Evolving Analysis Strategy")
    print(f"{'─' * 45}")
    print(f"  Dataset:    {args.data}")
    print(f"  Task:       {args.task}")
    print(f"  Provider:   {cfg.provider}")
    print(f"  Iterations: {args.iterations}")
    print(f"  Threshold:  {args.threshold}")
    print(f"{'─' * 45}\n")

    final_state = run_evomind(
        dataset_path=args.data,
        task_description=args.task,
        max_iterations=args.iterations,
        score_threshold=args.threshold,
        llm=llm,
        memory=memory,
        tracker=tracker,
    )

    # Print evolution history
    print("\n📊 Evolution History:")
    print(f"{'─' * 80}")
    for entry in final_state["history"]:
        strat = entry["strategy"]
        ev = entry["evaluation"]
        mutation = strat.get("mutation_applied", "initial")
        gen = strat.get("generation", 0)
        print(
            f"  [iter {entry['iteration']}] "
            f"gen={gen} mutation={mutation!r:15s} "
            f"strategy={strat['name']!r:30s} "
            f"score={ev['score']:.4f}"
        )
        if cfg.verbose:
            print(f"           steps={strat['steps']}")
            print(f"           rationale={ev.get('rationale', '')[:80]}")

    print(f"\n{'─' * 80}")
    print(f"🏁 Stopped: {final_state['stop_reason']}")
    print(f"🏆 Best strategy: {final_state['best_strategy']['name']} (score={final_state['best_score']:.4f})")
    print(f"   Steps: {final_state['best_strategy']['steps']}")
    print(f"   Params: {json.dumps(final_state['best_strategy'].get('params', {}))}")
    print(f"   Generation: {final_state['best_strategy'].get('generation', 0)}")

    # Multi-objective breakdown
    if "evaluation" in final_state:
        ev = final_state["evaluation"]
        if "insight_depth" in ev:
            print(f"\n   📈 Multi-objective fitness:")
            print(f"      Insight depth: {ev.get('insight_depth', 0):.2f}")
            print(f"      Coverage:      {ev.get('coverage', 0):.2f}")
            print(f"      Efficiency:    {ev.get('efficiency', 0):.2f}")
            print(f"      Novelty:       {ev.get('novelty', 0):.2f}")

    # Save results
    if args.save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = RESULTS_DIR / f"run_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
        serializable = {
            "task_description": final_state["task_description"],
            "dataset_path": final_state["dataset_path"],
            "dataset_summary": final_state.get("dataset_summary"),
            "task_signature": final_state.get("task_signature"),
            "history": final_state["history"],
            "best_strategy": final_state["best_strategy"],
            "best_score": final_state["best_score"],
            "stop_reason": final_state["stop_reason"],
            "telemetry": tracker.summary(),
        }
        out_path.write_text(json.dumps(serializable, indent=2))
        print(f"\n💾 Saved results to {out_path}")

    # Generate report
    if args.report:
        report = generate_run_report(
            final_state,
            tracker=tracker,
            output_dir=str(RESULTS_DIR),
        )
        print(f"\n📝 Report generated")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
