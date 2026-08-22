"""Multi-format data loader for EvoMind.

Supports CSV, Parquet, JSON/JSONL, Excel, TSV, and SQL databases.
Returns a pandas DataFrame regardless of source format.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


class DataLoadError(Exception):
    """Raised when a dataset cannot be loaded."""


def load_dataset(source: str, **kwargs: Any) -> pd.DataFrame:
    """Load a dataset from any supported source.

    Args:
        source: File path or connection string.
            Supported file extensions: .csv, .tsv, .parquet, .json, .jsonl, .xlsx, .xls
            For SQL: use 'sqlite:///path/to/db?table=tablename' or similar SQLAlchemy URI.
        **kwargs: Extra arguments passed through to the underlying pandas reader.

    Returns:
        A pandas DataFrame.

    Raises:
        DataLoadError: If the source format is unsupported or loading fails.
    """
    # --- SQL databases ---
    if source.startswith(("sqlite:///", "postgresql://", "mysql://", "mssql://")):
        return _load_sql(source, **kwargs)

    # --- HTTP/S URLs ---
    if source.startswith(("http://", "https://")):
        return _load_url(source, **kwargs)

    # --- Local files ---
    path = Path(source)
    if not path.exists():
        raise DataLoadError(f"File not found: {source}")

    ext = path.suffix.lower()
    loaders = {
        ".csv": _load_csv,
        ".tsv": _load_tsv,
        ".parquet": _load_parquet,
        ".json": _load_json,
        ".jsonl": _load_jsonl,
        ".xlsx": _load_excel,
        ".xls": _load_excel,
    }

    loader = loaders.get(ext)
    if loader is None:
        supported = ", ".join(sorted(loaders.keys()))
        raise DataLoadError(f"Unsupported file format: {ext}. Supported: {supported}")

    try:
        return loader(path, **kwargs)
    except DataLoadError:
        raise
    except Exception as exc:
        raise DataLoadError(f"Failed to load {source}: {exc}") from exc


# ---------------------------------------------------------------------------
# Format-specific loaders
# ---------------------------------------------------------------------------


def _load_csv(path: Path, **kwargs: Any) -> pd.DataFrame:
    return pd.read_csv(path, **kwargs)


def _load_tsv(path: Path, **kwargs: Any) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", **kwargs)


def _load_parquet(path: Path, **kwargs: Any) -> pd.DataFrame:
    return pd.read_parquet(path, **kwargs)


def _load_json(path: Path, **kwargs: Any) -> pd.DataFrame:
    # Try records orientation first, then let pandas figure it out
    try:
        return pd.read_json(path, **kwargs)
    except ValueError:
        return pd.read_json(path, lines=True, **kwargs)


def _load_jsonl(path: Path, **kwargs: Any) -> pd.DataFrame:
    return pd.read_json(path, lines=True, **kwargs)


def _load_excel(path: Path, **kwargs: Any) -> pd.DataFrame:
    return pd.read_excel(path, **kwargs)


def _load_sql(connection_string: str, **kwargs: Any) -> pd.DataFrame:
    """Load from a SQL database via SQLAlchemy."""
    try:
        from sqlalchemy import create_engine
    except ImportError as exc:
        raise DataLoadError("SQL support requires sqlalchemy: pip install sqlalchemy") from exc

    table_name = kwargs.pop("table", None)
    query = kwargs.pop("query", None)

    # Parse table from query param if in connection string
    if "?" in connection_string and table_name is None:
        base, qs = connection_string.rsplit("?", 1)
        for param in qs.split("&"):
            if param.startswith("table="):
                table_name = param.split("=", 1)[1]
                connection_string = base
                break

    engine = create_engine(connection_string)

    if query:
        return pd.read_sql_query(query, engine, **kwargs)
    elif table_name:
        return pd.read_sql_table(table_name, engine, **kwargs)
    else:
        raise DataLoadError("SQL source requires either 'table' or 'query' parameter")


def _load_url(url: str, **kwargs: Any) -> pd.DataFrame:
    """Load a dataset from a URL (CSV, JSON, Parquet)."""
    url_lower = url.lower()
    if url_lower.endswith(".parquet"):
        return pd.read_parquet(url, **kwargs)
    elif url_lower.endswith(".json") or url_lower.endswith(".jsonl"):
        return pd.read_json(url, **kwargs)
    else:
        # Default to CSV for URLs
        return pd.read_csv(url, **kwargs)


def validate_dataset(df: pd.DataFrame, max_mb: int = 500) -> None:
    """Check that a loaded dataset is safe to analyze."""
    mem_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
    if mem_mb > max_mb:
        raise DataLoadError(
            f"Dataset too large: {mem_mb:.1f} MB exceeds {max_mb} MB limit. "
            f"Sample it down or increase EVOMIND_MAX_DATASET_MB."
        )
    if df.shape[0] == 0:
        raise DataLoadError("Dataset is empty (0 rows).")
    if df.shape[1] == 0:
        raise DataLoadError("Dataset has no columns.")
