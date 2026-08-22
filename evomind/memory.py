"""Persistent memory of strategies EvoMind has tried, across runs and datasets.

This is what makes the "retains what works" part real: strategies that score
well on a given task signature are recalled and used as a starting point the
next time a similar task comes in, instead of the agent starting from zero.

v2 additions:
 - lineage tracking (parent_id, mutation_type, generation)
 - embedding-based cross-task similarity search
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from evomind.state import Strategy

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "evomind.db"

SCHEMA_BASE = """
CREATE TABLE IF NOT EXISTS strategies (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_signature TEXT NOT NULL,
    name TEXT NOT NULL,
    strategy_json TEXT NOT NULL,
    score REAL NOT NULL,
    rationale TEXT,
    parent_id INTEGER,
    mutation_type TEXT,
    generation INTEGER DEFAULT 0,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (parent_id) REFERENCES strategies(id)
);
CREATE INDEX IF NOT EXISTS idx_strategies_signature ON strategies (task_signature);

CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_signature TEXT NOT NULL,
    iteration INTEGER NOT NULL,
    strategy_json TEXT NOT NULL,
    score REAL NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS task_embeddings (
    task_signature TEXT PRIMARY KEY,
    task_description TEXT NOT NULL,
    embedding_json TEXT NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
"""

# Index that depends on columns added during migration — run AFTER _migrate()
SCHEMA_POST_MIGRATE = """
CREATE INDEX IF NOT EXISTS idx_strategies_parent ON strategies (parent_id);
"""


class StrategyMemory:
    """Thin wrapper around a SQLite DB storing strategies and their scores.

    Usable as a context manager, or call .close() when done. Safe to point
    multiple short-lived instances at the same file — each call opens and
    commits its own connection.
    """

    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _init_schema(self) -> None:
        with self._connect() as conn:
            # 1. Create tables (IF NOT EXISTS — safe on both old and new DBs)
            conn.executescript(SCHEMA_BASE)
            # 2. Add missing columns to old databases
            self._migrate(conn)
            # 3. Create indexes that depend on migrated columns
            conn.executescript(SCHEMA_POST_MIGRATE)

    @staticmethod
    def _migrate(conn: sqlite3.Connection) -> None:
        """Add columns/tables that may be missing from an older schema."""
        cursor = conn.execute("PRAGMA table_info(strategies)")
        existing_cols = {row[1] for row in cursor.fetchall()}

        migrations = [
            ("parent_id", "ALTER TABLE strategies ADD COLUMN parent_id INTEGER"),
            ("mutation_type", "ALTER TABLE strategies ADD COLUMN mutation_type TEXT"),
            ("generation", "ALTER TABLE strategies ADD COLUMN generation INTEGER DEFAULT 0"),
        ]
        for col_name, sql in migrations:
            if col_name not in existing_cols:
                try:
                    conn.execute(sql)
                except sqlite3.OperationalError:
                    pass
        conn.commit()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Core CRUD
    # ------------------------------------------------------------------

    def save_strategy(
        self,
        task_signature: str,
        strategy: Strategy,
        score: float,
        rationale: str = "",
        parent_id: int | None = None,
        mutation_type: str | None = None,
        generation: int = 0,
    ) -> int:
        """Persist a strategy + its score. Every attempt is stored (not just the best)."""
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO strategies "
                "(task_signature, name, strategy_json, score, rationale, parent_id, mutation_type, generation) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    task_signature,
                    strategy["name"],
                    json.dumps(strategy),
                    score,
                    rationale,
                    parent_id,
                    mutation_type,
                    generation,
                ),
            )
            return cur.lastrowid

    def record_run(self, task_signature: str, iteration: int, strategy: Strategy, score: float) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO runs (task_signature, iteration, strategy_json, score) VALUES (?, ?, ?, ?)",
                (task_signature, iteration, json.dumps(strategy), score),
            )

    def get_best_strategies(self, task_signature: str, top_k: int = 3) -> list[dict[str, Any]]:
        """Best-scoring strategies previously seen for this task signature."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, name, strategy_json, score, rationale FROM strategies "
                "WHERE task_signature = ? ORDER BY score DESC LIMIT ?",
                (task_signature, top_k),
            ).fetchall()
        return [
            {"id": r[0], "name": r[1], "strategy": json.loads(r[2]), "score": r[3], "rationale": r[4]}
            for r in rows
        ]

    def get_global_best(self, top_k: int = 5) -> list[dict[str, Any]]:
        """Best strategies across ALL task signatures — useful as general priors."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT task_signature, name, strategy_json, score FROM strategies "
                "ORDER BY score DESC LIMIT ?",
                (top_k,),
            ).fetchall()
        return [
            {"task_signature": r[0], "name": r[1], "strategy": json.loads(r[2]), "score": r[3]}
            for r in rows
        ]

    def history_for(self, task_signature: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT iteration, strategy_json, score FROM runs "
                "WHERE task_signature = ? ORDER BY iteration ASC",
                (task_signature,),
            ).fetchall()
        return [{"iteration": r[0], "strategy": json.loads(r[1]), "score": r[2]} for r in rows]

    # ------------------------------------------------------------------
    # Lineage / genealogy
    # ------------------------------------------------------------------

    def get_lineage(self, strategy_id: int) -> list[dict[str, Any]]:
        """Walk back the parent chain to build the full genealogy of a strategy."""
        lineage: list[dict[str, Any]] = []
        current_id: int | None = strategy_id
        with self._connect() as conn:
            while current_id is not None:
                row = conn.execute(
                    "SELECT id, name, score, parent_id, mutation_type, generation "
                    "FROM strategies WHERE id = ?",
                    (current_id,),
                ).fetchone()
                if row is None:
                    break
                lineage.append({
                    "id": row[0],
                    "name": row[1],
                    "score": row[2],
                    "parent_id": row[3],
                    "mutation_type": row[4],
                    "generation": row[5],
                })
                current_id = row[3]
        return lineage

    def get_evolution_tree(self, task_signature: str) -> list[dict[str, Any]]:
        """All strategies for a task signature with their parent links — for visualization."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, name, score, parent_id, mutation_type, generation "
                "FROM strategies WHERE task_signature = ? ORDER BY id ASC",
                (task_signature,),
            ).fetchall()
        return [
            {
                "id": r[0], "name": r[1], "score": r[2],
                "parent_id": r[3], "mutation_type": r[4], "generation": r[5],
            }
            for r in rows
        ]

    # ------------------------------------------------------------------
    # Embedding-based cross-task recall
    # ------------------------------------------------------------------

    def save_task_embedding(self, task_signature: str, task_description: str, embedding: list[float]) -> None:
        """Store an embedding vector for a task description."""
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO task_embeddings (task_signature, task_description, embedding_json) "
                "VALUES (?, ?, ?)",
                (task_signature, task_description, json.dumps(embedding)),
            )

    def find_similar_tasks(self, query_embedding: list[float], top_k: int = 3, exclude_signature: str | None = None) -> list[dict[str, Any]]:
        """Find tasks with the most similar embeddings (cosine similarity)."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT task_signature, task_description, embedding_json FROM task_embeddings"
            ).fetchall()

        if not rows:
            return []

        import math

        def cosine_sim(a: list[float], b: list[float]) -> float:
            dot = sum(x * y for x, y in zip(a, b))
            norm_a = math.sqrt(sum(x * x for x in a))
            norm_b = math.sqrt(sum(x * x for x in b))
            if norm_a == 0 or norm_b == 0:
                return 0.0
            return dot / (norm_a * norm_b)

        scored = []
        for r in rows:
            sig, desc, emb_json = r
            if sig == exclude_signature:
                continue
            emb = json.loads(emb_json)
            sim = cosine_sim(query_embedding, emb)
            scored.append({"task_signature": sig, "task_description": desc, "similarity": sim})

        scored.sort(key=lambda x: -x["similarity"])
        return scored[:top_k]

    def get_strategies_for_similar_tasks(
        self,
        query_embedding: list[float],
        top_k_tasks: int = 3,
        top_k_strategies: int = 2,
        exclude_signature: str | None = None,
    ) -> list[dict[str, Any]]:
        """Cross-task transfer: find similar tasks, then get their best strategies."""
        similar_tasks = self.find_similar_tasks(
            query_embedding, top_k=top_k_tasks, exclude_signature=exclude_signature
        )
        results: list[dict[str, Any]] = []
        for task in similar_tasks:
            strategies = self.get_best_strategies(task["task_signature"], top_k=top_k_strategies)
            for s in strategies:
                s["source_task"] = task["task_description"]
                s["similarity"] = task["similarity"]
                results.append(s)
        return results
