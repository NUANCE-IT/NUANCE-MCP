# Copyright © 2026 by Northwestern University. All Rights Reserved.
# SPDX-License-Identifier: MIT
# Terms of use: See LICENSE and TERMS_OF_USE.md for permitted use, disclaimers, and limitations.

"""Persistent experiment memory ("brain") — local, structured, no cloud.

This is episodic + semantic memory, not reinforcement learning: there is no
reward signal or policy here, only a durable log of what was run and a way
to recall it. :class:`~nuance_mcp.core.lifecycle.JobRegistry` is *working*
memory (current live jobs, lost on restart); :mod:`nuance_mcp.core.skills`
is *procedural* memory (the protocols themselves). This module adds the
missing *episodic* layer (what happened, on which instrument, with which
parameters) and a thin *semantic* query (`recall_similar`) over it, so a
skill can ask "what did we do last time on this material?" before acting.

Storage is a single local SQLite file — no vector DB, no external service,
consistent with the project's local-first, zero-cloud-dependency posture.
Microscopy experiment metadata (material, skill, parameters) is structured
enough that exact/LIKE filtering on indexed columns is sufficient; semantic
embedding search would be over-engineering for this data shape.

A skill does not call tools directly (see :mod:`nuance_mcp.core.skills`), so
memory is threaded through the same way every other effect is: the agent
calls ``start_experiment`` / ``log_tool_call`` / ``finish_experiment`` /
``recall_similar_experiments`` as ordinary tools, following steps written
into the skill's unrolled text. This module does not modify any skill body;
wiring specific skills to call these tools is a follow-up.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Optional


def default_db_path() -> str:
    """Resolve the default memory DB path.

    ``NUANCE_MEMORY_DB`` wins if set (``:memory:`` is valid, for tests and
    ephemeral simulator runs). Otherwise a file under ``~/.nuance_mcp/``,
    created on first use.
    """
    env = os.environ.get("NUANCE_MEMORY_DB")
    if env:
        return env
    data_dir = Path.home() / ".nuance_mcp"
    data_dir.mkdir(parents=True, exist_ok=True)
    return str(data_dir / "memory.sqlite3")


_SCHEMA = """
CREATE TABLE IF NOT EXISTS experiments (
    id              TEXT PRIMARY KEY,
    skill_name      TEXT NOT NULL,
    material        TEXT,
    arguments       TEXT NOT NULL,
    adapter_vendor  TEXT,
    adapter_model   TEXT,
    status          TEXT NOT NULL,
    summary         TEXT,
    started_at      REAL NOT NULL,
    finished_at     REAL
);
CREATE INDEX IF NOT EXISTS idx_experiments_material ON experiments(material);
CREATE INDEX IF NOT EXISTS idx_experiments_skill ON experiments(skill_name);

CREATE TABLE IF NOT EXISTS tool_calls (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id     TEXT NOT NULL REFERENCES experiments(id),
    seq               INTEGER NOT NULL,
    tool_name         TEXT NOT NULL,
    arguments         TEXT,
    result_summary    TEXT,
    at                REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_tool_calls_experiment ON tool_calls(experiment_id);
"""


class MemoryStore:
    """Thread-safe SQLite-backed episodic memory of past experiments.

    Mirrors the shape of :class:`~nuance_mcp.core.lifecycle.JobRegistry`
    (lock-guarded dict-like access) but persists to disk instead of
    living only in-process.
    """

    def __init__(self, db_path: Optional[str] = None) -> None:
        self.db_path = db_path or default_db_path()
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    # --- writes ----------------------------------------------------------

    def start_experiment(
        self,
        skill_name: str,
        arguments: dict[str, Any],
        material: Optional[str] = None,
        adapter_vendor: Optional[str] = None,
        adapter_model: Optional[str] = None,
    ) -> str:
        exp_id = f"exp-{uuid.uuid4().hex[:8]}"
        with self._lock:
            self._conn.execute(
                """INSERT INTO experiments
                   (id, skill_name, material, arguments, adapter_vendor,
                    adapter_model, status, summary, started_at, finished_at)
                   VALUES (?, ?, ?, ?, ?, ?, 'running', NULL, ?, NULL)""",
                (
                    exp_id,
                    skill_name,
                    material,
                    json.dumps(arguments),
                    adapter_vendor,
                    adapter_model,
                    time.time(),
                ),
            )
            self._conn.commit()
        return exp_id

    def log_tool_call(
        self,
        experiment_id: str,
        tool_name: str,
        arguments: Optional[dict[str, Any]] = None,
        result_summary: Optional[str] = None,
    ) -> None:
        with self._lock:
            (seq,) = self._conn.execute(
                "SELECT COUNT(*) FROM tool_calls WHERE experiment_id = ?",
                (experiment_id,),
            ).fetchone()
            self._conn.execute(
                """INSERT INTO tool_calls
                   (experiment_id, seq, tool_name, arguments, result_summary, at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    experiment_id,
                    seq,
                    tool_name,
                    json.dumps(arguments) if arguments is not None else None,
                    result_summary,
                    time.time(),
                ),
            )
            self._conn.commit()

    def finish_experiment(
        self,
        experiment_id: str,
        status: str = "ok",
        summary: Optional[str] = None,
    ) -> dict[str, Any]:
        with self._lock:
            self._conn.execute(
                """UPDATE experiments SET status = ?, summary = ?, finished_at = ?
                   WHERE id = ?""",
                (status, summary, time.time(), experiment_id),
            )
            self._conn.commit()
        return self.get_experiment(experiment_id)

    # --- reads -------------------------------------------------------------

    def get_experiment(self, experiment_id: str) -> dict[str, Any]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM experiments WHERE id = ?", (experiment_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"unknown experiment_id: {experiment_id}")
            calls = self._conn.execute(
                """SELECT tool_name, arguments, result_summary, at
                   FROM tool_calls WHERE experiment_id = ? ORDER BY seq""",
                (experiment_id,),
            ).fetchall()
        return self._row_to_dict(row, calls)

    def recent_experiments(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM experiments ORDER BY started_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def recall_similar(
        self,
        material: Optional[str] = None,
        skill_name: Optional[str] = None,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Recall past experiments matching material and/or skill.

        Exact/prefix filtering on indexed columns — deliberately not
        semantic/embedding search; see module docstring.
        """
        clauses, params = [], []
        if material:
            clauses.append("material = ?")
            params.append(material)
        if skill_name:
            clauses.append("skill_name = ?")
            params.append(skill_name)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._lock:
            rows = self._conn.execute(
                f"""SELECT * FROM experiments {where}
                    ORDER BY started_at DESC LIMIT ?""",
                (*params, limit),
            ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # --- internals -----------------------------------------------------

    @staticmethod
    def _row_to_dict(
        row: sqlite3.Row, calls: Optional[list[sqlite3.Row]] = None
    ) -> dict[str, Any]:
        out = {
            "id": row["id"],
            "skill_name": row["skill_name"],
            "material": row["material"],
            "arguments": json.loads(row["arguments"]),
            "adapter_vendor": row["adapter_vendor"],
            "adapter_model": row["adapter_model"],
            "status": row["status"],
            "summary": row["summary"],
            "started_at": row["started_at"],
            "finished_at": row["finished_at"],
        }
        if calls is not None:
            out["tool_calls"] = [
                {
                    "tool_name": c["tool_name"],
                    "arguments": json.loads(c["arguments"]) if c["arguments"] else None,
                    "result_summary": c["result_summary"],
                    "at": c["at"],
                }
                for c in calls
            ]
        return out


def register_memory(mcp, store: MemoryStore) -> None:
    """Register memory tools + resources on a FastMCP instance.

    Tools (called by the agent, same convention as skill-driven tool
    calls — see :mod:`nuance_mcp.core.skills`):
        start_experiment, log_tool_call, finish_experiment,
        recall_similar_experiments

    Resources (readable context, no explicit call required by the agent):
        memory://experiments/recent
        memory://experiments/{experiment_id}
    """

    @mcp.tool(name="start_experiment")
    def start_experiment(
        skill_name: str,
        arguments: dict[str, Any],
        material: str = "",
        adapter_vendor: str = "",
        adapter_model: str = "",
    ) -> dict:
        return {
            "experiment_id": store.start_experiment(
                skill_name,
                arguments,
                material or None,
                adapter_vendor or None,
                adapter_model or None,
            )
        }

    @mcp.tool(name="log_tool_call")
    def log_tool_call(
        experiment_id: str,
        tool_name: str,
        arguments: Optional[dict[str, Any]] = None,
        result_summary: str = "",
    ) -> dict:
        store.log_tool_call(experiment_id, tool_name, arguments, result_summary or None)
        return {"status": "logged"}

    @mcp.tool(name="finish_experiment")
    def finish_experiment(
        experiment_id: str, status: str = "ok", summary: str = ""
    ) -> dict:
        return store.finish_experiment(experiment_id, status, summary or None)

    @mcp.tool(name="recall_similar_experiments")
    def recall_similar_experiments(
        material: str = "", skill_name: str = "", limit: int = 5
    ) -> list[dict]:
        return store.recall_similar(material or None, skill_name or None, limit)

    @mcp.resource("memory://experiments/recent")
    def recent_experiments() -> list[dict]:
        return store.recent_experiments()

    @mcp.resource("memory://experiments/{experiment_id}")
    def experiment_by_id(experiment_id: str) -> dict:
        return store.get_experiment(experiment_id)
