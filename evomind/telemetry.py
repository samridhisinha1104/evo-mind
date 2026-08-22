"""Structured logging and telemetry for EvoMind.

Provides JSON-structured logging, LLM call tracing, timing, and
auto-generated run summary reports.
"""

from __future__ import annotations

import json
import logging
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


# ---------------------------------------------------------------------------
# JSON logger
# ---------------------------------------------------------------------------

_logger = logging.getLogger("evomind")


def setup_logging(verbose: bool = False) -> None:
    """Configure the evomind logger with structured JSON output."""
    level = logging.DEBUG if verbose else logging.INFO
    handler = logging.StreamHandler()
    handler.setFormatter(_JsonFormatter())
    _logger.handlers.clear()
    _logger.addHandler(handler)
    _logger.setLevel(level)
    _logger.propagate = False


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "msg": record.getMessage(),
        }
        if hasattr(record, "extra_data"):
            entry["data"] = record.extra_data  # type: ignore[attr-defined]
        return json.dumps(entry)


def log_event(event: str, **data: Any) -> None:
    """Log a structured event with optional key-value data."""
    record = _logger.makeRecord(
        "evomind", logging.INFO, "", 0, event, (), None
    )
    record.extra_data = data  # type: ignore[attr-defined]
    _logger.handle(record)


def log_debug(event: str, **data: Any) -> None:
    """Log a debug-level structured event."""
    record = _logger.makeRecord(
        "evomind", logging.DEBUG, "", 0, event, (), None
    )
    record.extra_data = data  # type: ignore[attr-defined]
    _logger.handle(record)


# ---------------------------------------------------------------------------
# Timing context manager
# ---------------------------------------------------------------------------

@contextmanager
def timed(label: str) -> Iterator[dict[str, float]]:
    """Context manager that records elapsed time and logs it."""
    timing: dict[str, float] = {}
    start = time.perf_counter()
    try:
        yield timing
    finally:
        elapsed = time.perf_counter() - start
        timing["elapsed_seconds"] = round(elapsed, 3)
        log_debug(f"timing:{label}", elapsed_seconds=timing["elapsed_seconds"])


# ---------------------------------------------------------------------------
# Run tracker
# ---------------------------------------------------------------------------

class RunTracker:
    """Tracks a full EvoMind run for telemetry and reporting."""

    def __init__(self) -> None:
        self.start_time = datetime.now(timezone.utc)
        self.events: list[dict[str, Any]] = []
        self.llm_calls: list[dict[str, Any]] = []
        self.timings: dict[str, float] = {}

    def record_event(self, event: str, **data: Any) -> None:
        self.events.append({
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": event,
            **data,
        })
        log_event(event, **data)

    def record_llm_call(
        self,
        provider: str,
        model: str,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        latency_ms: float | None = None,
        cost_usd: float | None = None,
    ) -> None:
        call = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "provider": provider,
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "latency_ms": latency_ms,
            "cost_usd": cost_usd,
        }
        self.llm_calls.append(call)

    def record_timing(self, label: str, seconds: float) -> None:
        self.timings[label] = round(seconds, 3)

    def summary(self) -> dict[str, Any]:
        """Generate a summary of the entire run."""
        end_time = datetime.now(timezone.utc)
        total_seconds = (end_time - self.start_time).total_seconds()

        total_llm_calls = len(self.llm_calls)
        total_cost = sum(c.get("cost_usd", 0) or 0 for c in self.llm_calls)
        avg_latency = (
            sum(c.get("latency_ms", 0) or 0 for c in self.llm_calls) / max(total_llm_calls, 1)
        )

        return {
            "start_time": self.start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "total_seconds": round(total_seconds, 2),
            "total_events": len(self.events),
            "llm_calls": total_llm_calls,
            "estimated_cost_usd": round(total_cost, 6),
            "avg_llm_latency_ms": round(avg_latency, 1),
            "timings": self.timings,
        }


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_run_report(
    final_state: dict[str, Any],
    tracker: RunTracker | None = None,
    output_dir: str | Path | None = None,
) -> str:
    """Generate a markdown report summarizing the run."""
    lines: list[str] = []
    lines.append("# EvoMind Run Report\n")
    lines.append(f"**Task:** {final_state.get('task_description', 'N/A')}\n")
    lines.append(f"**Dataset:** {final_state.get('dataset_path', 'N/A')}\n")
    lines.append(f"**Stop reason:** {final_state.get('stop_reason', 'N/A')}\n")

    if tracker:
        summary = tracker.summary()
        lines.append(f"\n## Telemetry\n")
        lines.append(f"- Total time: {summary['total_seconds']}s")
        lines.append(f"- LLM calls: {summary['llm_calls']}")
        lines.append(f"- Estimated cost: ${summary['estimated_cost_usd']}")
        lines.append(f"- Avg LLM latency: {summary['avg_llm_latency_ms']}ms\n")

    # Evolution history
    history = final_state.get("history", [])
    if history:
        lines.append("\n## Evolution History\n")
        lines.append("| Iter | Strategy | Score | Mutation | Rationale |")
        lines.append("|------|----------|-------|----------|-----------|")
        for entry in history:
            s = entry["strategy"]
            e = entry["evaluation"]
            mutation = s.get("mutation_applied", "initial")
            lines.append(
                f"| {entry['iteration']} | {s['name']} | {e['score']:.4f} | "
                f"{mutation} | {e.get('rationale', '')[:60]} |"
            )

    # Best strategy
    best = final_state.get("best_strategy")
    if best:
        lines.append(f"\n## Best Strategy: {best['name']} (score={final_state.get('best_score', 0):.4f})\n")
        lines.append(f"**Steps:** {', '.join(best.get('steps', []))}\n")
        lines.append(f"**Params:** `{json.dumps(best.get('params', {}))}`\n")
        lines.append(f"**Generation:** {best.get('generation', 0)}\n")

    report = "\n".join(lines)

    if output_dir:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        report_path = out_dir / f"report_{ts}.md"
        report_path.write_text(report, encoding="utf-8")
        log_event("report_saved", path=str(report_path))

    return report
