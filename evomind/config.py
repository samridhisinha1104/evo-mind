"""Centralized configuration for EvoMind.

Loads settings from environment variables, .env files, or evomind.toml.
Validates at startup so failures happen early, not mid-run.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


@dataclass
class EvoMindConfig:
    """All tuneable knobs in one place."""

    # --- LLM ---
    provider: str = "groq"
    model: str | None = None  # None = use provider default
    max_tokens: int = 2048
    temperature: float = 0.4

    # --- Evolution loop ---
    max_iterations: int = 5
    score_threshold: float = 0.8

    # --- Safety rails ---
    step_timeout_seconds: int = 30
    max_dataset_mb: int = 500

    # --- Paths ---
    db_path: str = str(Path(__file__).resolve().parent.parent / "data" / "evomind.db")
    results_dir: str = str(Path(__file__).resolve().parent.parent / "experiments" / "results")

    # --- Telemetry ---
    verbose: bool = False
    save_results: bool = False
    generate_report: bool = False

    # --- Data loading ---
    supported_formats: list[str] = field(default_factory=lambda: [".csv", ".parquet", ".json", ".jsonl", ".xlsx", ".tsv"])


def load_config(**overrides: object) -> EvoMindConfig:
    """Build config from env vars + overrides.  .env is loaded first."""
    load_dotenv()

    cfg = EvoMindConfig(
        provider=str(overrides.get("provider", os.environ.get("EVOMIND_PROVIDER", "groq"))),
        model=overrides.get("model") or os.environ.get("EVOMIND_MODEL") or None,  # type: ignore[arg-type]
        max_tokens=int(overrides.get("max_tokens", os.environ.get("EVOMIND_MAX_TOKENS", 2048))),
        temperature=float(overrides.get("temperature", os.environ.get("EVOMIND_TEMPERATURE", 0.4))),
        max_iterations=int(overrides.get("max_iterations", os.environ.get("EVOMIND_MAX_ITERATIONS", 5))),
        score_threshold=float(overrides.get("score_threshold", os.environ.get("EVOMIND_SCORE_THRESHOLD", 0.8))),
        step_timeout_seconds=int(overrides.get("step_timeout_seconds", os.environ.get("EVOMIND_STEP_TIMEOUT", 30))),
        max_dataset_mb=int(overrides.get("max_dataset_mb", os.environ.get("EVOMIND_MAX_DATASET_MB", 500))),
        verbose=bool(overrides.get("verbose", os.environ.get("EVOMIND_VERBOSE", "").lower() in ("1", "true"))),
        save_results=bool(overrides.get("save_results", False)),
        generate_report=bool(overrides.get("generate_report", False)),
    )

    if overrides.get("db_path"):
        cfg.db_path = str(overrides["db_path"])
    elif db_env := os.environ.get("EVOMIND_DB_PATH"):
        cfg.db_path = db_env

    if overrides.get("results_dir"):
        cfg.results_dir = str(overrides["results_dir"])

    return cfg
