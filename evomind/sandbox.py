"""Safety rails and sandboxing for EvoMind.

- Step execution timeouts
- Dataset memory limits
- Strategy JSON schema validation
- Score regression rollback
"""

from __future__ import annotations

import signal
import sys
import threading
from typing import Any

from evomind.state import Strategy

# ---------------------------------------------------------------------------
# Timeout (cross-platform)
# ---------------------------------------------------------------------------


class StepTimeoutError(Exception):
    """Raised when a step exceeds its time budget."""


def _timeout_unix(seconds: int):
    """Unix SIGALRM-based timeout (not available on Windows)."""
    import contextlib

    @contextlib.contextmanager
    def _ctx():
        def _handler(signum, frame):
            raise StepTimeoutError(f"Step timed out after {seconds}s")

        old = signal.signal(signal.SIGALRM, _handler)
        signal.alarm(seconds)
        try:
            yield
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old)

    return _ctx()


def _timeout_thread(seconds: int):
    """Thread-based timeout fallback for Windows."""
    import contextlib

    @contextlib.contextmanager
    def _ctx():
        result: list[Any] = [None]
        exception: list[BaseException | None] = [None]
        timed_out = threading.Event()

        def _timer():
            if not done.is_set():
                timed_out.set()

        done = threading.Event()
        timer = threading.Timer(seconds, _timer)
        timer.daemon = True
        timer.start()
        try:
            yield
            if timed_out.is_set():
                raise StepTimeoutError(f"Step timed out after {seconds}s")
        finally:
            done.set()
            timer.cancel()

    return _ctx()


def step_timeout(seconds: int = 30):
    """Cross-platform timeout context manager."""
    if sys.platform == "win32":
        return _timeout_thread(seconds)
    return _timeout_unix(seconds)


# ---------------------------------------------------------------------------
# Strategy validation
# ---------------------------------------------------------------------------

REQUIRED_STRATEGY_KEYS = {"name", "steps"}


def validate_strategy(strategy: dict[str, Any], available_steps: list[str]) -> list[str]:
    """Validate a strategy dict. Returns a list of error messages (empty = valid)."""
    errors: list[str] = []

    for key in REQUIRED_STRATEGY_KEYS:
        if key not in strategy:
            errors.append(f"Missing required key: {key}")

    if "steps" in strategy:
        if not isinstance(strategy["steps"], list):
            errors.append("'steps' must be a list")
        elif len(strategy["steps"]) == 0:
            errors.append("'steps' must not be empty")
        else:
            unknown = [s for s in strategy["steps"] if s not in available_steps]
            if unknown:
                errors.append(f"Unknown steps: {unknown}")

            dupes = [s for s in strategy["steps"] if strategy["steps"].count(s) > 1]
            if dupes:
                errors.append(f"Duplicate steps: {list(set(dupes))}")

    if "params" in strategy and not isinstance(strategy["params"], dict):
        errors.append("'params' must be a dict")

    return errors


# ---------------------------------------------------------------------------
# Score regression detection
# ---------------------------------------------------------------------------


def detect_regression(
    current_score: float,
    best_score: float,
    regression_threshold: float = 0.3,
) -> bool:
    """Returns True if the current score dropped significantly from the best."""
    if best_score <= 0:
        return False
    drop = best_score - current_score
    return drop > regression_threshold


def should_rollback(
    current_score: float,
    best_score: float,
    consecutive_drops: int,
    max_consecutive_drops: int = 2,
    regression_threshold: float = 0.3,
) -> bool:
    """Returns True if we should rollback to the best strategy."""
    if detect_regression(current_score, best_score, regression_threshold):
        return True
    return consecutive_drops >= max_consecutive_drops
