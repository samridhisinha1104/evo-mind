import numpy as np
import pandas as pd

from evomind.graph import build_graph, run_evomind
from evomind.memory import StrategyMemory


class ScriptedLLM:
    """Feeds fixed planner/evaluator responses in the order they're requested.

    Alternates: plan, eval, plan, eval, ... so the graph runs deterministically
    without ever touching the network.
    """

    def __init__(self, plans, evals):
        self.plans = list(plans)
        self.evals = list(evals)

    def complete_json(self, system, prompt):
        if "evaluation module" in system:
            return self.evals.pop(0)
        # Planner or mutation selector
        if "mutation" in system.lower() or "operator" in prompt.lower():
            # For mutation iterations, return operator selection
            return self.plans.pop(0)
        return self.plans.pop(0)


def make_df():
    rng = np.random.default_rng(1)
    n = 300
    a = rng.normal(size=n)
    return pd.DataFrame({"a": a, "b": a * 0.9 + rng.normal(scale=0.1, size=n), "c": rng.normal(size=n)})


def test_graph_stops_at_score_threshold(tmp_path):
    df = make_df()
    memory = StrategyMemory(tmp_path / "m.db")
    llm = ScriptedLLM(
        plans=[
            {"name": "first_pass", "steps": ["describe_data", "correlation_analysis"], "params": {"threshold": 0.5}},
        ],
        evals=[
            {"score": 0.95, "rationale": "found the strong a/b correlation immediately"},
        ],
    )

    compiled = build_graph(df, llm=llm, memory=memory)
    initial_state = {
        "task_description": "find useful patterns",
        "dataset_path": "unused.csv",
        "dataset_summary": {"n_numeric": 3, "n_rows": 300, "n_cols": 3, "n_missing_cells": 0, "columns": {}},
        "task_signature": "many_numeric_large_rows",
        "history": [],
        "iteration": 0,
        "max_iterations": 5,
        "score_threshold": 0.7,
        "best_score": -1.0,
        "should_continue": True,
    }
    final_state = compiled.invoke(initial_state, config={"recursion_limit": 30})

    assert final_state["stop_reason"] == "score_threshold_reached"
    assert len(final_state["history"]) == 1
    assert final_state["best_strategy"]["name"] == "first_pass"


def test_graph_loops_until_max_iterations_then_keeps_best(tmp_path):
    df = make_df()
    memory = StrategyMemory(tmp_path / "m.db")
    llm = ScriptedLLM(
        plans=[
            {"name": "iter0", "steps": ["describe_data"], "params": {}},
            # iter1: mutation operator selection, then applied deterministically
            {"operators": ["add_step"], "reasoning": "need more steps"},
            # iter2: mutation operator selection
            {"operators": ["swap_step"], "reasoning": "try different approach"},
        ],
        evals=[
            {"score": 0.2, "rationale": "weak"},
            {"score": 0.6, "rationale": "better"},
            {"score": 0.4, "rationale": "worse than iter1"},
        ],
    )

    compiled = build_graph(df, llm=llm, memory=memory)
    initial_state = {
        "task_description": "find useful patterns",
        "dataset_path": "unused.csv",
        "dataset_summary": {"n_numeric": 3, "n_rows": 300, "n_cols": 3, "n_missing_cells": 0, "columns": {}},
        "task_signature": "many_numeric_large_rows_2",
        "history": [],
        "iteration": 0,
        "max_iterations": 3,
        "score_threshold": 0.99,
        "best_score": -1.0,
        "should_continue": True,
    }
    final_state = compiled.invoke(initial_state, config={"recursion_limit": 30})

    assert final_state["stop_reason"] == "max_iterations_reached"
    assert len(final_state["history"]) == 3
    # best score should be retained even though the last iteration was worse
    assert final_state["best_score"] > 0.3


def test_run_evomind_end_to_end_with_real_csv(tmp_path):
    csv_path = tmp_path / "data.csv"
    make_df().to_csv(csv_path, index=False)

    llm = ScriptedLLM(
        plans=[{"name": "one_shot", "steps": ["describe_data", "correlation_analysis"], "params": {"threshold": 0.5}}],
        evals=[{"score": 1.0, "rationale": "done"}],
    )
    memory = StrategyMemory(tmp_path / "m.db")

    final_state = run_evomind(
        dataset_path=str(csv_path),
        task_description="find useful patterns",
        max_iterations=3,
        score_threshold=0.7,
        llm=llm,
        memory=memory,
    )

    assert final_state["stop_reason"] == "score_threshold_reached"
    assert final_state["dataset_summary"]["n_rows"] == 300


def test_run_evomind_with_parquet(tmp_path):
    """Test that the multi-format loader works with Parquet files."""
    parquet_path = tmp_path / "data.parquet"
    make_df().to_parquet(parquet_path, index=False)

    llm = ScriptedLLM(
        plans=[{"name": "parquet_test", "steps": ["describe_data"], "params": {}}],
        evals=[{"score": 1.0, "rationale": "done"}],
    )
    memory = StrategyMemory(tmp_path / "m.db")

    final_state = run_evomind(
        dataset_path=str(parquet_path),
        task_description="test parquet loading",
        max_iterations=1,
        score_threshold=0.5,
        llm=llm,
        memory=memory,
    )

    assert final_state["dataset_summary"]["n_rows"] == 300


def test_run_evomind_with_json(tmp_path):
    """Test JSON data loading."""
    json_path = tmp_path / "data.json"
    make_df().to_json(json_path, orient="records")

    llm = ScriptedLLM(
        plans=[{"name": "json_test", "steps": ["describe_data"], "params": {}}],
        evals=[{"score": 1.0, "rationale": "done"}],
    )
    memory = StrategyMemory(tmp_path / "m.db")

    final_state = run_evomind(
        dataset_path=str(json_path),
        task_description="test json loading",
        max_iterations=1,
        score_threshold=0.5,
        llm=llm,
        memory=memory,
    )

    assert final_state["dataset_summary"]["n_rows"] == 300
