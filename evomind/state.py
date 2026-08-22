"""Shared state passed between every node in the EvoMind graph.

Everything nodes read or write lives here. Keep it flat and JSON-serialisable
so it can be logged, checkpointed, or persisted without extra work.
"""

from __future__ import annotations

from typing import Any, TypedDict


class Strategy(TypedDict):
    """A concrete plan of analysis steps the executor can run."""

    name: str
    description: str
    steps: list[str]          # ordered, human-readable step identifiers
    params: dict[str, Any]     # step-specific knobs (e.g. n_clusters, iqr_multiplier)
    # --- lineage (optional, added by mutation system) ---
    parent_name: str           # which strategy was this mutated from?
    mutation_applied: str      # which operator(s) were used?
    generation: int            # how many mutations deep are we?


class StepResult(TypedDict):
    step: str
    summary: str
    data: dict[str, Any]
    error: str | None


class Evaluation(TypedDict):
    score: float               # 0-1, composite score (backward compat)
    insight_depth: float       # 0-1: how non-obvious are the findings?
    coverage: float            # 0-1: what % of the dataset was analyzed?
    efficiency: float          # 0-1: inverse of redundant/failed steps
    novelty: float             # 0-1: how different from prior strategies?
    rationale: str
    signal_count: int          # how many non-trivial findings were produced


class EvoState(TypedDict, total=False):
    # --- task setup (fixed for the run) ---
    task_description: str
    dataset_path: str
    dataset_summary: dict[str, Any]
    task_signature: str        # short hash/key used to look up prior strategies in memory

    # --- evolving loop state ---
    strategy: Strategy
    step_results: list[StepResult]
    evaluation: Evaluation
    history: list[dict[str, Any]]   # one entry per iteration: {strategy, evaluation}
    iteration: int
    max_iterations: int
    score_threshold: float

    # --- control flow ---
    best_strategy: Strategy
    best_score: float
    should_continue: bool
    stop_reason: str
