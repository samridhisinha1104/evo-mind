import numpy as np
import pandas as pd
import pytest

from evomind.memory import StrategyMemory
from evomind.nodes import (
    AVAILABLE_STEPS,
    evaluator_node,
    executor_node,
    make_task_signature,
    planner_node,
    reflector_node,
    summarize_dataset,
)


class FakeLLM:
    """Deterministic stand-in for any LLM client — no network, no API key."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def complete_json(self, system, prompt):
        self.calls.append((system, prompt))
        if not self.responses:
            raise RuntimeError("FakeLLM ran out of scripted responses")
        return self.responses.pop(0)


@pytest.fixture
def sample_df():
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "a": rng.normal(size=200),
            "b": rng.normal(size=200) * 2 + 1,
            "cat": rng.choice(["x", "y", "z"], size=200),
        }
    )


def test_summarize_dataset(sample_df):
    summary = summarize_dataset(sample_df)
    assert summary["n_rows"] == 200
    assert summary["n_numeric"] == 2
    assert summary["n_categorical"] == 1
    assert "memory_mb" in summary


def test_make_task_signature_buckets():
    small = {"n_numeric": 2, "n_rows": 50, "n_cols": 3, "n_missing_cells": 0, "columns": {}}
    large = {"n_numeric": 10, "n_rows": 5000, "n_cols": 12, "n_missing_cells": 0, "columns": {}}
    sig_small = make_task_signature("t", small)
    sig_large = make_task_signature("t", large)
    # New signatures encode dtype profile + row bucket + missingness + task hash
    assert "num2_" in sig_small
    assert "tiny" in sig_small  # 50 rows = tiny
    assert "num10_" in sig_large
    assert "medium" in sig_large  # 5000 rows = medium
    assert "clean" in sig_small  # 0 missing = clean
    # Different datasets should produce different signatures
    assert sig_small != sig_large


def test_planner_node_first_iteration_uses_llm_proposal(tmp_path):
    llm = FakeLLM([{"name": "custom", "description": "d", "steps": ["describe_data", "correlation_analysis"], "params": {}}])
    memory = StrategyMemory(tmp_path / "m.db")
    state = {
        "task_description": "find patterns",
        "dataset_summary": {"n_numeric": 2, "n_rows": 200, "n_cols": 3, "n_missing_cells": 0, "columns": {}},
        "task_signature": "few_numeric_small_rows",
        "iteration": 0,
    }
    out = planner_node(state, llm=llm, memory=memory)
    assert out["strategy"]["name"] == "custom"
    assert set(out["strategy"]["steps"]).issubset(set(AVAILABLE_STEPS))
    # Lineage should be set
    assert out["strategy"]["generation"] == 0


def test_planner_node_falls_back_on_llm_failure(tmp_path):
    class BrokenLLM:
        def complete_json(self, system, prompt):
            raise ValueError("boom")

    state = {
        "task_description": "find patterns",
        "dataset_summary": {"n_numeric": 2, "n_rows": 200, "n_cols": 3, "n_missing_cells": 0, "columns": {}},
        "task_signature": "few_numeric_small_rows",
        "iteration": 0,
    }
    out = planner_node(state, llm=BrokenLLM(), memory=StrategyMemory(tmp_path / "m.db"))
    assert out["strategy"]["steps"]  # non-empty fallback


def test_executor_node_runs_all_steps_and_survives_bad_step(sample_df):
    state = {
        "strategy": {
            "name": "s",
            "description": "",
            "steps": ["describe_data", "correlation_analysis", "not_a_real_step"],
            "params": {},
            "parent_name": "",
            "mutation_applied": "",
            "generation": 0,
        }
    }
    out = executor_node(state, df=sample_df)
    results = out["step_results"]
    assert len(results) == 3
    assert results[0]["error"] is None
    assert results[2]["error"] == "unknown_step"


def test_executor_runs_new_steps(sample_df):
    """Verify the new step implementations (Phase 2.3) actually run."""
    new_steps = [
        "data_profiling",
        "feature_importance_rf",
        "anomaly_detection_iforest",
        "mutual_information",
        "distribution_analysis",
        "dbscan_clustering",
        "time_series_decomposition",
    ]
    state = {
        "strategy": {
            "name": "new_steps_test",
            "description": "",
            "steps": new_steps,
            "params": {},
            "parent_name": "",
            "mutation_applied": "",
            "generation": 0,
        }
    }
    out = executor_node(state, df=sample_df)
    results = out["step_results"]
    assert len(results) == len(new_steps)
    # All steps should succeed (no errors)
    for r in results:
        assert r["error"] is None, f"Step {r['step']} failed: {r['error']}"


def test_evaluator_node_blends_heuristic_and_llm_score(sample_df):
    state = {
        "task_description": "find patterns",
        "strategy": {
            "name": "s", "description": "", "steps": ["correlation_analysis"], "params": {},
            "parent_name": "", "mutation_applied": "", "generation": 0,
        },
        "step_results": executor_node(
            {"strategy": {"name": "s", "steps": ["correlation_analysis"], "params": {},
                          "parent_name": "", "mutation_applied": "", "generation": 0}},
            df=sample_df,
        )["step_results"],
        "history": [],
    }
    llm = FakeLLM([{"score": 0.9, "rationale": "solid correlation signal"}])
    out = evaluator_node(state, llm=llm)
    assert 0.0 <= out["evaluation"]["score"] <= 1.0
    assert out["evaluation"]["rationale"] == "solid correlation signal"
    # Multi-objective dimensions should be present
    assert "insight_depth" in out["evaluation"]
    assert "coverage" in out["evaluation"]
    assert "efficiency" in out["evaluation"]
    assert "novelty" in out["evaluation"]


def test_evaluator_does_not_give_perfect_score_on_baseline(sample_df):
    """The bug we fixed: baseline strategy should NOT score 1.0 on first iteration."""
    state = {
        "task_description": "find patterns",
        "strategy": {
            "name": "baseline",
            "description": "",
            "steps": ["describe_data", "missing_value_report", "correlation_analysis", "outlier_detection_iqr"],
            "params": {"threshold": 0.6, "iqr_multiplier": 1.5},
            "parent_name": "", "mutation_applied": "", "generation": 0,
        },
        "history": [],
    }
    state["step_results"] = executor_node(state, df=sample_df)["step_results"]

    # Use a fake LLM that gives a moderate score
    llm = FakeLLM([{"score": 0.6, "rationale": "decent but could explore more"}])
    out = evaluator_node(state, llm=llm)
    # Should NOT be 1.0 anymore — the fixed heuristic caps at 0.92
    assert out["evaluation"]["score"] < 1.0, f"Score should not be 1.0, got {out['evaluation']['score']}"


def test_reflector_node_stops_at_threshold(tmp_path):
    memory = StrategyMemory(tmp_path / "m.db")
    state = {
        "task_signature": "sig",
        "strategy": {
            "name": "s", "description": "", "steps": [], "params": {},
            "parent_name": "", "mutation_applied": "", "generation": 0,
        },
        "evaluation": {
            "score": 0.95, "rationale": "great", "signal_count": 5,
            "insight_depth": 0.8, "coverage": 0.9, "efficiency": 1.0, "novelty": 0.5,
        },
        "history": [],
        "iteration": 0,
        "score_threshold": 0.8,
        "max_iterations": 5,
        "best_score": -1.0,
    }
    out = reflector_node(state, memory=memory)
    assert out["should_continue"] is False
    assert out["stop_reason"] == "score_threshold_reached"
    assert out["best_score"] == 0.95


def test_reflector_node_continues_when_below_threshold_and_iterations_remain(tmp_path):
    memory = StrategyMemory(tmp_path / "m.db")
    state = {
        "task_signature": "sig",
        "strategy": {
            "name": "s", "description": "", "steps": [], "params": {},
            "parent_name": "", "mutation_applied": "", "generation": 0,
        },
        "evaluation": {
            "score": 0.3, "rationale": "meh", "signal_count": 1,
            "insight_depth": 0.1, "coverage": 0.2, "efficiency": 0.5, "novelty": 1.0,
        },
        "history": [],
        "iteration": 0,
        "score_threshold": 0.8,
        "max_iterations": 5,
        "best_score": -1.0,
    }
    out = reflector_node(state, memory=memory)
    assert out["should_continue"] is True
    assert out["iteration"] == 1


def test_reflector_node_stops_at_max_iterations(tmp_path):
    memory = StrategyMemory(tmp_path / "m.db")
    state = {
        "task_signature": "sig",
        "strategy": {
            "name": "s", "description": "", "steps": [], "params": {},
            "parent_name": "", "mutation_applied": "", "generation": 0,
        },
        "evaluation": {
            "score": 0.3, "rationale": "meh", "signal_count": 1,
            "insight_depth": 0.1, "coverage": 0.2, "efficiency": 0.5, "novelty": 1.0,
        },
        "history": [],
        "iteration": 4,
        "score_threshold": 0.8,
        "max_iterations": 5,
        "best_score": -1.0,
    }
    out = reflector_node(state, memory=memory)
    assert out["should_continue"] is False
    assert out["stop_reason"] == "max_iterations_reached"
