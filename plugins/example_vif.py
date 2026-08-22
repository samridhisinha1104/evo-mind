"""Example EvoMind plugin: adds a Variance Inflation Factor (VIF) step.

Drop this file in the `plugins/` directory and it will be auto-loaded.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from evomind.state import StepResult


def _step_vif_analysis(df: pd.DataFrame, params: dict) -> StepResult:
    """Compute Variance Inflation Factor to detect multicollinearity."""
    from numpy.linalg import LinAlgError

    num = df.select_dtypes(include=[np.number]).dropna()
    if num.shape[1] < 2 or num.shape[0] < 10:
        return StepResult(
            step="vif_analysis",
            summary="Not enough data for VIF",
            data={},
            error=None,
        )

    vif_scores = {}
    X = num.values
    for i, col in enumerate(num.columns):
        try:
            # VIF = 1 / (1 - R^2)
            others = np.delete(X, i, axis=1)
            corr_matrix = np.corrcoef(X[:, i], others, rowvar=False)
            r_squared = corr_matrix[0, 1:] ** 2
            max_r2 = float(np.nanmax(r_squared)) if len(r_squared) > 0 else 0.0
            vif = 1.0 / (1.0 - max_r2) if max_r2 < 1.0 else float("inf")
            vif_scores[col] = round(vif, 3)
        except (LinAlgError, ValueError):
            continue

    high_vif = {k: v for k, v in vif_scores.items() if v > params.get("vif_threshold", 5.0)}

    return StepResult(
        step="vif_analysis",
        summary=f"{len(high_vif)} features with high multicollinearity (VIF>{params.get('vif_threshold', 5.0)})",
        data={"vif_scores": vif_scores, "high_vif": high_vif},
        error=None,
    )


def register(registry):
    """Called automatically by the plugin loader."""
    registry.register_step(
        name="vif_analysis",
        fn=_step_vif_analysis,
        description="Compute Variance Inflation Factor to detect multicollinearity",
        required_dtypes=["numeric"],
    )
