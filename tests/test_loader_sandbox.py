"""Tests for multi-format data loader and sandbox safety."""

import pandas as pd
import numpy as np
import pytest

from evomind.loader import DataLoadError, load_dataset, validate_dataset
from evomind.sandbox import detect_regression, should_rollback, validate_strategy


# ---- Data loader tests ----

def test_load_csv(tmp_path):
    csv = tmp_path / "test.csv"
    pd.DataFrame({"a": [1, 2], "b": [3, 4]}).to_csv(csv, index=False)
    df = load_dataset(str(csv))
    assert len(df) == 2
    assert list(df.columns) == ["a", "b"]


def test_load_tsv(tmp_path):
    tsv = tmp_path / "test.tsv"
    pd.DataFrame({"a": [1, 2], "b": [3, 4]}).to_csv(tsv, sep="\t", index=False)
    df = load_dataset(str(tsv))
    assert len(df) == 2


def test_load_parquet(tmp_path):
    pq = tmp_path / "test.parquet"
    pd.DataFrame({"a": [1, 2], "b": [3, 4]}).to_parquet(pq, index=False)
    df = load_dataset(str(pq))
    assert len(df) == 2


def test_load_json(tmp_path):
    jf = tmp_path / "test.json"
    pd.DataFrame({"a": [1, 2], "b": [3, 4]}).to_json(jf, orient="records")
    df = load_dataset(str(jf))
    assert len(df) == 2


def test_load_jsonl(tmp_path):
    jl = tmp_path / "test.jsonl"
    pd.DataFrame({"a": [1, 2], "b": [3, 4]}).to_json(jl, orient="records", lines=True)
    df = load_dataset(str(jl))
    assert len(df) == 2


def test_load_file_not_found():
    with pytest.raises(DataLoadError, match="not found"):
        load_dataset("/nonexistent/file.csv")


def test_load_unsupported_format(tmp_path):
    bad = tmp_path / "test.xyz"
    bad.write_text("data")
    with pytest.raises(DataLoadError, match="Unsupported"):
        load_dataset(str(bad))


def test_validate_dataset_empty():
    with pytest.raises(DataLoadError, match="empty"):
        validate_dataset(pd.DataFrame())


def test_validate_dataset_too_large():
    # Create a DataFrame that reports > 1 MB
    df = pd.DataFrame({"a": ["x" * 1000] * 1100})
    with pytest.raises(DataLoadError, match="too large"):
        validate_dataset(df, max_mb=1)


# ---- Sandbox / safety tests ----

def test_validate_strategy_valid():
    strategy = {"name": "test", "steps": ["describe_data"], "params": {}}
    errors = validate_strategy(strategy, ["describe_data", "correlation_analysis"])
    assert errors == []


def test_validate_strategy_missing_name():
    strategy = {"steps": ["describe_data"]}
    errors = validate_strategy(strategy, ["describe_data"])
    assert any("name" in e for e in errors)


def test_validate_strategy_empty_steps():
    strategy = {"name": "test", "steps": []}
    errors = validate_strategy(strategy, ["describe_data"])
    assert any("empty" in e for e in errors)


def test_validate_strategy_unknown_steps():
    strategy = {"name": "test", "steps": ["describe_data", "fake_step"]}
    errors = validate_strategy(strategy, ["describe_data"])
    assert any("Unknown" in e for e in errors)


def test_validate_strategy_duplicate_steps():
    strategy = {"name": "test", "steps": ["describe_data", "describe_data"]}
    errors = validate_strategy(strategy, ["describe_data"])
    assert any("Duplicate" in e for e in errors)


def test_detect_regression():
    assert detect_regression(0.3, 0.8, 0.3) is True
    assert detect_regression(0.7, 0.8, 0.3) is False
    assert detect_regression(0.5, 0.0, 0.3) is False  # no best yet


def test_should_rollback():
    assert should_rollback(0.2, 0.8, 0, regression_threshold=0.3) is True
    assert should_rollback(0.7, 0.8, 3, max_consecutive_drops=2) is True
    assert should_rollback(0.7, 0.8, 1, max_consecutive_drops=2) is False
