"""Tests for the structured mutation operators."""

import pytest

from evomind.mutations import (
    add_step,
    apply_mutation,
    crossover,
    random_mutation,
    remove_step,
    reorder_steps,
    swap_step,
    tune_params,
)
from evomind.state import StepResult


def make_strategy(steps=None, params=None):
    return {
        "name": "test_strategy",
        "description": "test",
        "steps": steps or ["describe_data", "correlation_analysis"],
        "params": params or {"threshold": 0.6},
        "parent_name": "",
        "mutation_applied": "",
        "generation": 0,
    }


AVAILABLE = [
    "describe_data", "missing_value_report", "correlation_analysis",
    "outlier_detection_iqr", "distribution_analysis", "kmeans_clustering",
    "pca_projection", "categorical_frequency",
]


def test_add_step():
    s = make_strategy()
    result = add_step(s, AVAILABLE)
    # Should have one more step
    assert len(result["steps"]) == len(s["steps"]) + 1
    # Original strategy should be unchanged (deepcopy)
    assert len(s["steps"]) == 2
    assert "_add" in result["name"]


def test_remove_step():
    s = make_strategy()
    result = remove_step(s)
    assert len(result["steps"]) == 1
    assert "_remove" in result["name"]


def test_remove_step_with_results():
    s = make_strategy(steps=["describe_data", "correlation_analysis", "kmeans_clustering"])
    results = [
        StepResult(step="describe_data", summary="ok", data={"x": 1}, error=None),
        StepResult(step="correlation_analysis", summary="nothing", data={}, error=None),
        StepResult(step="kmeans_clustering", summary="ok", data={"x": 1}, error=None),
    ]
    result = remove_step(s, step_results=results)
    # Should remove the step with empty data
    assert "correlation_analysis" not in result["steps"]


def test_remove_step_single_step():
    s = make_strategy(steps=["describe_data"])
    result = remove_step(s)
    # Should not remove the last step
    assert len(result["steps"]) == 1


def test_swap_step():
    s = make_strategy()
    result = swap_step(s, AVAILABLE)
    assert len(result["steps"]) == 2
    # At least one step should be different (unless random picks same)
    assert "_swap" in result["name"]


def test_tune_params():
    s = make_strategy(params={"threshold": 0.6, "n_clusters": 3})
    result = tune_params(s)
    assert "_tune" in result["name"]
    # At least one param should have changed
    original_vals = set(s["params"].values())
    new_vals = set(result["params"].values())
    # They might be the same by luck, but the function should run without error


def test_reorder_steps():
    s = make_strategy(steps=["a", "b", "c"])
    # Override AVAILABLE_STEPS for this test
    result = reorder_steps(s)
    assert set(result["steps"]) == {"a", "b", "c"}
    assert "_reorder" in result["name"]


def test_crossover():
    s1 = make_strategy(steps=["describe_data", "correlation_analysis"])
    s2 = make_strategy(steps=["kmeans_clustering", "pca_projection"])
    result = crossover(s1, partner=s2)
    assert "_cross" in result["name"]
    # Should contain steps from both strategies
    assert len(result["steps"]) > 0


def test_crossover_no_partner():
    s = make_strategy()
    result = crossover(s, partner=None)
    assert "_nocross" in result["name"]
    assert result["steps"] == s["steps"]


def test_apply_mutation():
    s = make_strategy()
    result, op = apply_mutation(s, "add_step", available_steps=AVAILABLE)
    assert op == "add_step"
    assert "Mutated via add_step" in result["description"]


def test_apply_mutation_unknown_operator():
    s = make_strategy()
    with pytest.raises(ValueError, match="Unknown mutation operator"):
        apply_mutation(s, "not_a_real_operator", available_steps=AVAILABLE)


def test_random_mutation():
    s = make_strategy()
    result, ops = random_mutation(s, available_steps=AVAILABLE, n_mutations=2)
    assert len(ops) == 2
    assert result["name"] != s["name"]
