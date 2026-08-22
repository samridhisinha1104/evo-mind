"""The four nodes of the EvoMind loop.

    planner   -> proposes/mutates a Strategy (a named, ordered list of analysis steps)
    executor  -> actually runs that strategy against the dataset with pandas/sklearn
    evaluator -> scores how useful the results were
    reflector -> decides: keep exploring (mutate) or stop, and updates memory

Every node is a plain function `def node(state: EvoState) -> dict` returning only
the keys it changes, which is what LangGraph expects for state updates.
"""

from __future__ import annotations

import hashlib
import json
import traceback
from typing import Any, Callable

import numpy as np
import pandas as pd

from evomind.llm import LLMClient, get_llm_client
from evomind.memory import StrategyMemory
from evomind.mutations import MUTATION_OPERATORS, apply_mutation, random_mutation
from evomind.state import Evaluation, EvoState, StepResult, Strategy


# --------------------------------------------------------------------------
# Registry of analysis steps the executor can run. The planner is constrained
# to pick names from this list, so every strategy it invents is guaranteed
# to be executable.
# --------------------------------------------------------------------------

AVAILABLE_STEPS = [
    # --- Original 8 ---
    "describe_data",
    "missing_value_report",
    "correlation_analysis",
    "outlier_detection_iqr",
    "distribution_analysis",
    "kmeans_clustering",
    "pca_projection",
    "categorical_frequency",
    # --- New 8 ---
    "data_profiling",
    "feature_importance_rf",
    "anomaly_detection_iforest",
    "mutual_information",
    "chi_squared_test",
    "group_comparison_ttest",
    "dbscan_clustering",
    "time_series_decomposition",
]


def _numeric_df(df: pd.DataFrame) -> pd.DataFrame:
    return df.select_dtypes(include=[np.number])


# --------------------------------------------------------------------------
# Original step implementations
# --------------------------------------------------------------------------


def _step_describe_data(df: pd.DataFrame, params: dict) -> StepResult:
    desc = df.describe(include="all").fillna("").to_dict()
    return StepResult(step="describe_data", summary=f"{df.shape[0]} rows, {df.shape[1]} cols", data={"describe": desc}, error=None)


def _step_missing_value_report(df: pd.DataFrame, params: dict) -> StepResult:
    missing = df.isna().sum()
    pct = (missing / len(df) * 100).round(2)
    report = {c: {"missing": int(missing[c]), "pct": float(pct[c])} for c in df.columns if missing[c] > 0}
    return StepResult(step="missing_value_report", summary=f"{len(report)} columns have missing values", data=report, error=None)


def _step_correlation_analysis(df: pd.DataFrame, params: dict) -> StepResult:
    num = _numeric_df(df)
    if num.shape[1] < 2:
        return StepResult(step="correlation_analysis", summary="fewer than 2 numeric columns", data={}, error=None)
    corr = num.corr(numeric_only=True)
    threshold = params.get("threshold", 0.6)
    strong_pairs = []
    cols = corr.columns
    for i, c1 in enumerate(cols):
        for c2 in cols[i + 1:]:
            v = corr.loc[c1, c2]
            if pd.notna(v) and abs(v) >= threshold:
                strong_pairs.append({"pair": [c1, c2], "corr": round(float(v), 3)})
    strong_pairs.sort(key=lambda p: -abs(p["corr"]))
    return StepResult(
        step="correlation_analysis",
        summary=f"{len(strong_pairs)} strongly correlated pairs (|r|>={threshold})",
        data={"strong_pairs": strong_pairs},
        error=None,
    )


def _step_outlier_detection_iqr(df: pd.DataFrame, params: dict) -> StepResult:
    num = _numeric_df(df)
    multiplier = params.get("iqr_multiplier", 1.5)
    outlier_counts = {}
    for col in num.columns:
        q1, q3 = num[col].quantile(0.25), num[col].quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - multiplier * iqr, q3 + multiplier * iqr
        n_outliers = int(((num[col] < lo) | (num[col] > hi)).sum())
        if n_outliers > 0:
            outlier_counts[col] = n_outliers
    return StepResult(
        step="outlier_detection_iqr",
        summary=f"{sum(outlier_counts.values())} outlier values across {len(outlier_counts)} columns",
        data={"outliers_by_column": outlier_counts},
        error=None,
    )


def _step_distribution_analysis(df: pd.DataFrame, params: dict) -> StepResult:
    num = _numeric_df(df)
    skewed = {}
    for col in num.columns:
        s = num[col].skew()
        if pd.notna(s) and abs(s) > params.get("skew_threshold", 1.0):
            skewed[col] = round(float(s), 3)
    return StepResult(
        step="distribution_analysis",
        summary=f"{len(skewed)} columns are notably skewed",
        data={"skewed_columns": skewed},
        error=None,
    )


def _step_kmeans_clustering(df: pd.DataFrame, params: dict) -> StepResult:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    num = _numeric_df(df).dropna()
    if num.shape[0] < 10 or num.shape[1] < 2:
        return StepResult(step="kmeans_clustering", summary="not enough numeric data to cluster", data={}, error=None)
    k = min(params.get("n_clusters", 3), max(2, num.shape[0] // 5))
    X = StandardScaler().fit_transform(num)
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
    sizes = pd.Series(km.labels_).value_counts().to_dict()
    return StepResult(
        step="kmeans_clustering",
        summary=f"{k} clusters found, sizes {sizes}",
        data={"k": k, "cluster_sizes": {int(k_): int(v) for k_, v in sizes.items()}, "inertia": float(km.inertia_)},
        error=None,
    )


def _step_pca_projection(df: pd.DataFrame, params: dict) -> StepResult:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    num = _numeric_df(df).dropna()
    if num.shape[1] < 2 or num.shape[0] < 5:
        return StepResult(step="pca_projection", summary="not enough numeric data for PCA", data={}, error=None)
    n_components = min(params.get("n_components", 2), num.shape[1])
    X = StandardScaler().fit_transform(num)
    pca = PCA(n_components=n_components).fit(X)
    explained = [round(float(v), 4) for v in pca.explained_variance_ratio_]
    return StepResult(
        step="pca_projection",
        summary=f"top {n_components} components explain {round(sum(explained) * 100, 1)}% of variance",
        data={"explained_variance_ratio": explained},
        error=None,
    )


def _step_categorical_frequency(df: pd.DataFrame, params: dict) -> StepResult:
    cat = df.select_dtypes(include=["object", "category"])
    top_n = params.get("top_n", 5)
    freqs = {col: cat[col].value_counts().head(top_n).to_dict() for col in cat.columns}
    return StepResult(step="categorical_frequency", summary=f"analyzed {len(freqs)} categorical columns", data=freqs, error=None)


# --------------------------------------------------------------------------
# New step implementations (Phase 2.3)
# --------------------------------------------------------------------------


def _step_data_profiling(df: pd.DataFrame, params: dict) -> StepResult:
    """Comprehensive data profiling: cardinality, uniqueness, dtypes, constants."""
    profile = {}
    for col in df.columns:
        n_unique = int(df[col].nunique())
        n_total = len(df)
        profile[col] = {
            "dtype": str(df[col].dtype),
            "n_unique": n_unique,
            "uniqueness_pct": round(n_unique / max(n_total, 1) * 100, 2),
            "is_constant": n_unique <= 1,
            "is_id_like": n_unique == n_total,
            "sample_values": [str(v) for v in df[col].dropna().head(3).tolist()],
        }
    n_constants = sum(1 for v in profile.values() if v["is_constant"])
    n_id_like = sum(1 for v in profile.values() if v["is_id_like"])
    return StepResult(
        step="data_profiling",
        summary=f"Profiled {len(profile)} columns: {n_constants} constants, {n_id_like} ID-like",
        data={"profile": profile},
        error=None,
    )


def _step_feature_importance_rf(df: pd.DataFrame, params: dict) -> StepResult:
    """Random Forest feature importance — tells the agent which columns matter."""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler

    num = _numeric_df(df).dropna()
    if num.shape[1] < 2 or num.shape[0] < 20:
        return StepResult(step="feature_importance_rf", summary="not enough data for RF", data={}, error=None)

    # Use the column with highest variance as target
    target_col = num.var().idxmax()
    X = num.drop(columns=[target_col])
    y = num[target_col]
    X_scaled = StandardScaler().fit_transform(X)

    rf = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42, n_jobs=-1)
    rf.fit(X_scaled, y)

    importances = dict(zip(X.columns, [round(float(v), 4) for v in rf.feature_importances_]))
    importances = dict(sorted(importances.items(), key=lambda x: -x[1]))

    return StepResult(
        step="feature_importance_rf",
        summary=f"Top feature: {list(importances.keys())[0]} (importance={list(importances.values())[0]}) predicting {target_col}",
        data={"importances": importances, "target": target_col, "r2_oob": round(float(rf.score(X_scaled, y)), 4)},
        error=None,
    )


def _step_anomaly_detection_iforest(df: pd.DataFrame, params: dict) -> StepResult:
    """Isolation Forest anomaly detection — finds multivariate outliers that IQR misses."""
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler

    num = _numeric_df(df).dropna()
    if num.shape[0] < 20 or num.shape[1] < 2:
        return StepResult(step="anomaly_detection_iforest", summary="not enough data for IsolationForest", data={}, error=None)

    contamination = params.get("contamination", 0.05)
    X = StandardScaler().fit_transform(num)
    iso = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)
    labels = iso.fit_predict(X)
    n_anomalies = int((labels == -1).sum())

    return StepResult(
        step="anomaly_detection_iforest",
        summary=f"{n_anomalies} anomalies detected ({round(n_anomalies / len(num) * 100, 1)}% of data)",
        data={"n_anomalies": n_anomalies, "contamination": contamination, "total_rows": len(num)},
        error=None,
    )


def _step_mutual_information(df: pd.DataFrame, params: dict) -> StepResult:
    """Mutual information — catches non-linear dependencies that correlation misses."""
    from sklearn.feature_selection import mutual_info_regression
    from sklearn.preprocessing import StandardScaler

    num = _numeric_df(df).dropna()
    if num.shape[1] < 2 or num.shape[0] < 20:
        return StepResult(step="mutual_information", summary="not enough data for MI", data={}, error=None)

    target_col = num.var().idxmax()
    X = num.drop(columns=[target_col])
    y = num[target_col]
    X_scaled = StandardScaler().fit_transform(X)

    mi_scores_raw = mutual_info_regression(X_scaled, y, random_state=42)
    mi_scores = dict(zip(X.columns, [round(float(v), 4) for v in mi_scores_raw]))
    mi_scores = dict(sorted(mi_scores.items(), key=lambda x: -x[1]))

    return StepResult(
        step="mutual_information",
        summary=f"MI scores computed for {len(mi_scores)} features vs {target_col}",
        data={"mi_scores": mi_scores, "target": target_col},
        error=None,
    )


def _step_chi_squared_test(df: pd.DataFrame, params: dict) -> StepResult:
    """Chi-squared test for categorical-to-categorical association."""
    from scipy.stats import chi2_contingency

    cat = df.select_dtypes(include=["object", "category"])
    if cat.shape[1] < 2:
        return StepResult(step="chi_squared_test", summary="fewer than 2 categorical columns", data={}, error=None)

    alpha = params.get("alpha", 0.05)
    significant_pairs = []
    cols = list(cat.columns)
    for i, c1 in enumerate(cols):
        for c2 in cols[i + 1:]:
            contingency = pd.crosstab(cat[c1], cat[c2])
            if contingency.shape[0] < 2 or contingency.shape[1] < 2:
                continue
            chi2, p_value, dof, _ = chi2_contingency(contingency)
            if p_value < alpha:
                significant_pairs.append({
                    "pair": [c1, c2],
                    "chi2": round(float(chi2), 3),
                    "p_value": round(float(p_value), 6),
                    "dof": int(dof),
                })
    significant_pairs.sort(key=lambda x: x["p_value"])

    return StepResult(
        step="chi_squared_test",
        summary=f"{len(significant_pairs)} significant categorical associations (p<{alpha})",
        data={"significant_pairs": significant_pairs},
        error=None,
    )


def _step_group_comparison_ttest(df: pd.DataFrame, params: dict) -> StepResult:
    """T-test: compare numeric distributions across groups defined by categorical columns."""
    from scipy.stats import ttest_ind

    cat = df.select_dtypes(include=["object", "category"])
    num = _numeric_df(df)
    if cat.shape[1] < 1 or num.shape[1] < 1:
        return StepResult(step="group_comparison_ttest", summary="need both categorical and numeric columns", data={}, error=None)

    alpha = params.get("alpha", 0.05)
    significant_results = []

    for cat_col in cat.columns:
        groups = df[cat_col].dropna().unique()
        if len(groups) != 2:
            continue  # t-test is for exactly 2 groups
        g1, g2 = groups[0], groups[1]
        for num_col in num.columns:
            a = df[df[cat_col] == g1][num_col].dropna()
            b = df[df[cat_col] == g2][num_col].dropna()
            if len(a) < 5 or len(b) < 5:
                continue
            stat, p_value = ttest_ind(a, b, equal_var=False)
            if p_value < alpha:
                significant_results.append({
                    "categorical": cat_col,
                    "numeric": num_col,
                    "groups": [str(g1), str(g2)],
                    "t_stat": round(float(stat), 3),
                    "p_value": round(float(p_value), 6),
                    "mean_diff": round(float(a.mean() - b.mean()), 4),
                })
    significant_results.sort(key=lambda x: x["p_value"])

    return StepResult(
        step="group_comparison_ttest",
        summary=f"{len(significant_results)} significant group differences found",
        data={"significant_results": significant_results},
        error=None,
    )


def _step_dbscan_clustering(df: pd.DataFrame, params: dict) -> StepResult:
    """DBSCAN clustering — finds non-spherical clusters that k-means misses."""
    from sklearn.cluster import DBSCAN
    from sklearn.preprocessing import StandardScaler

    num = _numeric_df(df).dropna()
    if num.shape[0] < 20 or num.shape[1] < 2:
        return StepResult(step="dbscan_clustering", summary="not enough data for DBSCAN", data={}, error=None)

    eps = params.get("eps", 0.5)
    min_samples = params.get("min_samples", 5)
    X = StandardScaler().fit_transform(num)
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    labels = pd.Series(db.labels_)
    n_clusters = int(labels[labels >= 0].nunique())
    n_noise = int((labels == -1).sum())
    sizes = labels[labels >= 0].value_counts().to_dict()

    return StepResult(
        step="dbscan_clustering",
        summary=f"DBSCAN: {n_clusters} clusters, {n_noise} noise points",
        data={
            "n_clusters": n_clusters,
            "n_noise": n_noise,
            "cluster_sizes": {int(k): int(v) for k, v in sizes.items()},
            "eps": eps,
            "min_samples": min_samples,
        },
        error=None,
    )


def _step_time_series_decomposition(df: pd.DataFrame, params: dict) -> StepResult:
    """Basic time-series decomposition: detect trends and seasonality."""
    num = _numeric_df(df)
    if num.shape[1] < 1 or num.shape[0] < 20:
        return StepResult(step="time_series_decomposition", summary="not enough data for TS decomposition", data={}, error=None)

    # Use the first numeric column as the time series
    col = num.columns[0]
    series = num[col].dropna().reset_index(drop=True)

    # Simple moving average trend
    window = min(params.get("window", 10), len(series) // 3)
    if window < 3:
        return StepResult(step="time_series_decomposition", summary="series too short for decomposition", data={}, error=None)

    trend = series.rolling(window=window, center=True).mean()
    residual = series - trend

    # Basic stationarity check via coefficient of variation of rolling mean
    rolling_mean = series.rolling(window=window).mean().dropna()
    cv = float(rolling_mean.std() / max(rolling_mean.mean(), 1e-10))

    return StepResult(
        step="time_series_decomposition",
        summary=f"TS decomposed: trend CV={round(cv, 3)}, {'non-stationary' if cv > 0.1 else 'roughly stationary'}",
        data={
            "column": col,
            "trend_cv": round(cv, 4),
            "is_stationary": cv <= 0.1,
            "window": window,
            "trend_mean": round(float(trend.mean()), 4) if not trend.isna().all() else None,
            "residual_std": round(float(residual.std()), 4) if not residual.isna().all() else None,
        },
        error=None,
    )


# --------------------------------------------------------------------------
# Step registry
# --------------------------------------------------------------------------

STEP_REGISTRY: dict[str, Callable[[pd.DataFrame, dict], StepResult]] = {
    # Original 8
    "describe_data": _step_describe_data,
    "missing_value_report": _step_missing_value_report,
    "correlation_analysis": _step_correlation_analysis,
    "outlier_detection_iqr": _step_outlier_detection_iqr,
    "distribution_analysis": _step_distribution_analysis,
    "kmeans_clustering": _step_kmeans_clustering,
    "pca_projection": _step_pca_projection,
    "categorical_frequency": _step_categorical_frequency,
    # New 8
    "data_profiling": _step_data_profiling,
    "feature_importance_rf": _step_feature_importance_rf,
    "anomaly_detection_iforest": _step_anomaly_detection_iforest,
    "mutual_information": _step_mutual_information,
    "chi_squared_test": _step_chi_squared_test,
    "group_comparison_ttest": _step_group_comparison_ttest,
    "dbscan_clustering": _step_dbscan_clustering,
    "time_series_decomposition": _step_time_series_decomposition,
}


# --------------------------------------------------------------------------
# Nodes
# --------------------------------------------------------------------------


def summarize_dataset(df: pd.DataFrame) -> dict[str, Any]:
    return {
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "columns": {c: str(t) for c, t in df.dtypes.items()},
        "n_numeric": int(_numeric_df(df).shape[1]),
        "n_categorical": int(df.select_dtypes(include=["object", "category"]).shape[1]),
        "n_missing_cells": int(df.isna().sum().sum()),
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2),
    }


def make_task_signature(task_description: str, dataset_summary: dict[str, Any]) -> str:
    """A discriminating key so similar tasks recall relevant strategies.

    Encodes dtype profile, row-count bucket, missingness level, and a hash
    of the task description keywords.  Much more precise than the old
    4-bucket system while still allowing structurally similar datasets to match.
    """
    n_num = dataset_summary.get("n_numeric", 0)
    n_rows = dataset_summary.get("n_rows", 0)
    n_cols = dataset_summary.get("n_cols", 0)
    n_cat = n_cols - n_num  # approximate categorical count

    # Detect datetime columns if available
    columns = dataset_summary.get("columns", {})
    n_datetime = sum(1 for dtype in columns.values() if "datetime" in str(dtype).lower())

    # Row bucket
    if n_rows < 100:
        row_bucket = "tiny"
    elif n_rows < 1_000:
        row_bucket = "small"
    elif n_rows < 10_000:
        row_bucket = "medium"
    elif n_rows < 100_000:
        row_bucket = "large"
    else:
        row_bucket = "huge"

    # Missingness level
    total_cells = max(n_rows * n_cols, 1)
    missing_pct = dataset_summary.get("n_missing_cells", 0) / total_cells
    if missing_pct < 0.01:
        miss = "clean"
    elif missing_pct < 0.10:
        miss = "some_missing"
    elif missing_pct < 0.30:
        miss = "messy"
    else:
        miss = "very_messy"

    # Task keyword hash (first 6 hex chars)
    task_words = sorted(set(task_description.lower().split()))
    task_hash = hashlib.md5(" ".join(task_words).encode()).hexdigest()[:6]

    return f"num{n_num}_cat{n_cat}_dt{n_datetime}_{row_bucket}_{miss}_{task_hash}"


DEFAULT_FALLBACK_STRATEGY: Strategy = {
    "name": "baseline_exploration",
    "description": "Generic first pass: describe, check quality, find relationships.",
    "steps": ["describe_data", "missing_value_report", "correlation_analysis", "outlier_detection_iqr"],
    "params": {"threshold": 0.6, "iqr_multiplier": 1.5},
    "parent_name": "",
    "mutation_applied": "",
    "generation": 0,
}


def planner_node(state: EvoState, llm: LLMClient | None = None, memory: StrategyMemory | None = None) -> dict:
    """Propose a strategy on iteration 0, or mutate via structured operators on later iterations.

    Iteration 0: LLM proposes an initial strategy seeded with recalled memory.
    Iteration 1+: LLM chooses a mutation operator, which is then applied deterministically.
    """
    llm = llm or get_llm_client()
    iteration = state.get("iteration", 0)

    if iteration == 0:
        memory = memory or StrategyMemory()
        recalled = memory.get_best_strategies(state["task_signature"], top_k=3)
        prompt = (
            f"Task: {state['task_description']}\n"
            f"Dataset summary: {json.dumps(state['dataset_summary'])}\n"
            f"Available analysis steps (use ONLY these names): {AVAILABLE_STEPS}\n"
            f"Previously successful strategies on similar tasks (may be empty): {json.dumps(recalled)}\n\n"
            "Propose an initial Strategy as JSON with keys: "
            "name (string), description (string), steps (ordered list of step names from the available list, no duplicates), "
            "params (object with any of: threshold, iqr_multiplier, skew_threshold, n_clusters, n_components, "
            "top_n, contamination, eps, min_samples, alpha, window)."
        )
        system = "You are the planning module of an agentic data-analysis system. Pick a sensible, minimal starting strategy."

        try:
            proposed = llm.complete_json(system, prompt)
            strategy: Strategy = {
                "name": proposed["name"],
                "description": proposed.get("description", ""),
                "steps": [s for s in proposed["steps"] if s in AVAILABLE_STEPS] or DEFAULT_FALLBACK_STRATEGY["steps"],
                "params": proposed.get("params", {}),
                "parent_name": "",
                "mutation_applied": "initial",
                "generation": 0,
            }
        except Exception:
            strategy = dict(DEFAULT_FALLBACK_STRATEGY)

    else:
        # --- Structured mutation via operator selection ---
        prev_strategy = state["strategy"]
        prev_results = state.get("step_results", [])

        # Ask LLM which mutation operator(s) to apply
        mutation_ops = list(MUTATION_OPERATORS.keys())
        prompt = (
            f"Task: {state['task_description']}\n"
            f"Current strategy: {json.dumps(prev_strategy)}\n"
            f"Its evaluation: {json.dumps(state['evaluation'])}\n"
            f"Step results summary: {json.dumps([r['summary'] for r in prev_results])}\n"
            f"Available mutation operators: {mutation_ops}\n"
            f"Available analysis steps: {AVAILABLE_STEPS}\n\n"
            "Choose 1-2 mutation operators to apply. Respond as JSON with keys: "
            "operators (list of operator names), reasoning (one sentence why)."
        )
        system = "You are the reflection/mutation module of an agentic data-analysis system. Choose which operator to apply to improve the strategy."

        # Get best partner from memory for crossover
        partner = None
        if memory:
            best = memory.get_best_strategies(state["task_signature"], top_k=1)
            if best and best[0]["strategy"]["name"] != prev_strategy["name"]:
                partner = best[0]["strategy"]

        try:
            llm_choice = llm.complete_json(system, prompt)
            chosen_ops = llm_choice.get("operators", ["add_step"])
            # Validate operators
            chosen_ops = [op for op in chosen_ops if op in MUTATION_OPERATORS]
            if not chosen_ops:
                chosen_ops = ["add_step"]

            # Apply operators sequentially
            strategy = prev_strategy
            applied_ops = []
            for op_name in chosen_ops[:2]:  # max 2 operators
                strategy, _ = apply_mutation(
                    strategy,
                    op_name,
                    available_steps=AVAILABLE_STEPS,
                    step_results=prev_results,
                    partner=partner,
                )
                applied_ops.append(op_name)

            strategy["parent_name"] = prev_strategy["name"]
            strategy["mutation_applied"] = "+".join(applied_ops)
            strategy["generation"] = prev_strategy.get("generation", 0) + 1

        except Exception:
            # Fallback: random mutation
            strategy, ops = random_mutation(
                prev_strategy,
                available_steps=AVAILABLE_STEPS,
                step_results=prev_results,
                partner=partner,
                n_mutations=1,
            )
            strategy["parent_name"] = prev_strategy["name"]
            strategy["mutation_applied"] = "+".join(ops)
            strategy["generation"] = prev_strategy.get("generation", 0) + 1

    return {"strategy": strategy}


def executor_node(state: EvoState, df: pd.DataFrame | None = None) -> dict:
    """Run every step in the strategy against the dataset, collecting results and errors."""
    if df is None:
        df = pd.read_csv(state["dataset_path"])

    results: list[StepResult] = []
    for step_name in state["strategy"]["steps"]:
        fn = STEP_REGISTRY.get(step_name)
        if fn is None:
            results.append(StepResult(step=step_name, summary="unknown step, skipped", data={}, error="unknown_step"))
            continue
        try:
            results.append(fn(df, state["strategy"].get("params", {})))
        except Exception as exc:  # keep going — one bad step shouldn't kill the run
            results.append(
                StepResult(step=step_name, summary="step failed", data={}, error=f"{exc}\n{traceback.format_exc(limit=2)}")
            )

    return {"step_results": results}


# Steps that represent deeper analytical work (used for depth bonus)
_DEEP_STEPS = {
    "kmeans_clustering", "pca_projection", "correlation_analysis",
    "feature_importance_rf", "anomaly_detection_iforest", "mutual_information",
    "dbscan_clustering", "chi_squared_test", "group_comparison_ttest",
    "time_series_decomposition",
}


def _heuristic_score(step_results: list[StepResult]) -> tuple[float, int]:
    """Deterministic scoring that forces genuine exploration.

    Uses signal *density* (per step, not raw count) to prevent instant
    saturation, rewards step diversity, and gives a depth bonus for
    non-trivial analytical steps.  Capped at 0.92 so only the LLM
    evaluator can award a perfect 1.0.
    """
    if not step_results:
        return 0.0, 0

    ok = [r for r in step_results if r["error"] is None]
    success_rate = len(ok) / max(len(step_results), 1)

    signal_count = 0
    for r in ok:
        data = r["data"]
        if not data:
            continue
        if r["step"] == "correlation_analysis":
            signal_count += len(data.get("strong_pairs", []))
        elif r["step"] == "outlier_detection_iqr":
            signal_count += len(data.get("outliers_by_column", {}))
        elif r["step"] == "distribution_analysis":
            signal_count += len(data.get("skewed_columns", {}))
        elif r["step"] == "missing_value_report":
            signal_count += len(data)
        elif r["step"] in ("kmeans_clustering", "pca_projection"):
            signal_count += 1
        elif r["step"] == "categorical_frequency":
            signal_count += len(data)
        elif r["step"] == "describe_data":
            signal_count += 1
        elif r["step"] == "feature_importance_rf":
            signal_count += len(data.get("importances", {}))
        elif r["step"] == "anomaly_detection_iforest":
            signal_count += 1 if data.get("n_anomalies", 0) > 0 else 0
        elif r["step"] == "mutual_information":
            signal_count += len(data.get("mi_scores", {}))
        elif r["step"] == "dbscan_clustering":
            signal_count += 1 if data.get("n_clusters", 0) > 1 else 0
        elif r["step"] == "chi_squared_test":
            signal_count += len(data.get("significant_pairs", []))
        elif r["step"] == "group_comparison_ttest":
            signal_count += len(data.get("significant_results", []))
        elif r["step"] == "time_series_decomposition":
            signal_count += 1 if data else 0
        elif r["step"] == "data_profiling":
            signal_count += 1
        else:
            # Unknown / plugin steps: count as 1 if they produced data
            signal_count += 1

    # --- Component scores ---

    # 1. Signal density: signals per successful step (prevents count-stuffing)
    signal_density = min(signal_count / (len(ok) * 3 + 1), 1.0)

    # 2. Step diversity: reward covering multiple analytical angles
    step_types_used = len(set(r["step"] for r in ok))
    diversity = min(step_types_used / 5.0, 1.0)

    # 3. Depth bonus: reward using non-trivial analytical steps
    has_deep = any(r["step"] in _DEEP_STEPS for r in ok)
    depth_bonus = 0.12 if has_deep else 0.0

    # --- Blend ---
    score = 0.25 * success_rate + 0.35 * signal_density + 0.20 * diversity + depth_bonus

    # Clamp: heuristic alone can never reach 1.0 — that requires LLM confirmation
    score = round(min(score, 0.92), 4)
    return score, signal_count


def _compute_multi_objective(
    step_results: list[StepResult],
    strategy: Strategy,
    history: list[dict[str, Any]],
    heuristic_score: float,
    signal_count: int,
) -> dict[str, float]:
    """Compute multi-objective fitness dimensions."""
    ok = [r for r in step_results if r["error"] is None]
    total = max(len(step_results), 1)

    # Insight depth: proportion of deep analytical steps that produced data
    deep_ok = [r for r in ok if r["step"] in _DEEP_STEPS and r["data"]]
    insight_depth = min(len(deep_ok) / 3.0, 1.0)

    # Coverage: how many different analytical angles were covered
    step_types = len(set(r["step"] for r in ok))
    coverage = min(step_types / 6.0, 1.0)

    # Efficiency: success rate penalized by failed/unknown steps
    efficiency = len(ok) / total

    # Novelty: how different is this strategy from past ones?
    novelty = 1.0
    if history:
        past_step_sets = [set(h["strategy"].get("steps", [])) for h in history]
        current_steps = set(strategy.get("steps", []))
        if past_step_sets:
            # Jaccard distance from most recent
            last = past_step_sets[-1]
            if current_steps or last:
                jaccard = len(current_steps & last) / max(len(current_steps | last), 1)
                novelty = round(1.0 - jaccard, 4)

    return {
        "insight_depth": round(insight_depth, 4),
        "coverage": round(coverage, 4),
        "efficiency": round(efficiency, 4),
        "novelty": round(novelty, 4),
    }


def evaluator_node(state: EvoState, llm: LLMClient | None = None) -> dict:
    """Score the run: deterministic heuristic + multi-objective dimensions, refined by LLM."""
    heuristic_score, signal_count = _heuristic_score(state["step_results"])
    rationale = f"Heuristic: signal_density+diversity+depth -> score={heuristic_score}"

    # Multi-objective dimensions
    multi_obj = _compute_multi_objective(
        state["step_results"],
        state["strategy"],
        state.get("history", []),
        heuristic_score,
        signal_count,
    )

    llm = llm or get_llm_client()
    try:
        prompt = (
            f"Task: {state['task_description']}\n"
            f"Strategy used: {json.dumps(state['strategy'])}\n"
            f"Step results: {json.dumps([{k: v for k, v in r.items() if k != 'data'} for r in state['step_results']])}\n"
            f"Heuristic score: {heuristic_score}\n"
            f"Multi-objective scores: {json.dumps(multi_obj)}\n\n"
            "Judge how useful these results actually are for the task. Respond as JSON with keys: "
            "score (float 0-1), rationale (one sentence)."
        )
        judged = llm.complete_json("You are the evaluation module of an agentic data-analysis system.", prompt)
        final_score = round(0.5 * heuristic_score + 0.5 * float(judged["score"]), 4)
        rationale = judged.get("rationale", rationale)
    except Exception:
        final_score = heuristic_score

    evaluation: Evaluation = {
        "score": final_score,
        "insight_depth": multi_obj["insight_depth"],
        "coverage": multi_obj["coverage"],
        "efficiency": multi_obj["efficiency"],
        "novelty": multi_obj["novelty"],
        "rationale": rationale,
        "signal_count": signal_count,
    }
    return {"evaluation": evaluation}


def reflector_node(state: EvoState, memory: StrategyMemory | None = None) -> dict:
    """Decide whether to keep evolving, update best-so-far, and persist to memory."""
    memory = memory or StrategyMemory()
    iteration = state.get("iteration", 0)
    evaluation = state["evaluation"]
    strategy = state["strategy"]

    # Save with lineage
    parent_name = strategy.get("parent_name", "")
    mutation_type = strategy.get("mutation_applied", "")
    generation = strategy.get("generation", 0)

    memory.save_strategy(
        state["task_signature"],
        strategy,
        evaluation["score"],
        evaluation["rationale"],
        mutation_type=mutation_type,
        generation=generation,
    )
    memory.record_run(state["task_signature"], iteration, strategy, evaluation["score"])

    history = list(state.get("history", []))
    history.append({"iteration": iteration, "strategy": strategy, "evaluation": evaluation})

    best_score = state.get("best_score", -1.0)
    best_strategy = state.get("best_strategy", strategy)
    if evaluation["score"] > best_score:
        best_score = evaluation["score"]
        best_strategy = strategy

    next_iteration = iteration + 1
    threshold = state.get("score_threshold", 0.8)
    max_iterations = state.get("max_iterations", 5)

    if evaluation["score"] >= threshold:
        should_continue, stop_reason = False, "score_threshold_reached"
    elif next_iteration >= max_iterations:
        should_continue, stop_reason = False, "max_iterations_reached"
    else:
        should_continue, stop_reason = True, ""

    return {
        "history": history,
        "best_score": best_score,
        "best_strategy": best_strategy,
        "iteration": next_iteration,
        "should_continue": should_continue,
        "stop_reason": stop_reason,
    }
