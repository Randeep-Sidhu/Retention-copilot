"""Run log and human review queue (SQLite). Separate from the read-only analytics database."""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from .config import RUNS_DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS agent_runs (
  run_id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT, customer_id TEXT, config TEXT, model TEXT,
  status TEXT, attempts INTEGER, llm_calls INTEGER, prompt_tokens INTEGER, completion_tokens INTEGER,
  llm_latency_s REAL, wall_s REAL, proposal_json TEXT, first_violations_json TEXT, violations_json TEXT,
  retrieved_ids_json TEXT, trace_json TEXT);
CREATE TABLE IF NOT EXISTS action_queue (
  queue_id INTEGER PRIMARY KEY AUTOINCREMENT, run_id INTEGER, customer_id TEXT, status TEXT DEFAULT 'pending',
  proposal_json TEXT, flags_json TEXT, violations_json TEXT, reviewer_note TEXT, created_at TEXT, decided_at TEXT);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class RunStore:
    def __init__(self, path: Path | str = RUNS_DB_PATH):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._con() as con:
            con.executescript(SCHEMA)

    @contextmanager
    def _con(self):
        con = sqlite3.connect(self.path)
        con.row_factory = sqlite3.Row
        try:
            yield con
            con.commit()
        finally:
            con.close()

    def save_run(self, rec: dict) -> int:
        with self._con() as con:
            cur = con.execute(
                "INSERT INTO agent_runs (created_at, customer_id, config, model, status, attempts, llm_calls, "
                "prompt_tokens, completion_tokens, llm_latency_s, wall_s, proposal_json, first_violations_json, "
                "violations_json, retrieved_ids_json, trace_json) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (_now(), rec["customer_id"], rec["config"], rec["model"], rec["status"], rec["attempts"],
                 rec["llm_calls"], rec["prompt_tokens"], rec["completion_tokens"], rec["llm_latency_s"],
                 rec["wall_s"], json.dumps(rec["proposal"]), json.dumps(rec["first_violations"]),
                 json.dumps(rec["violations"]), json.dumps(rec["retrieved_ids"]), json.dumps(rec["trace"])))
            return cur.lastrowid

    def enqueue(self, run_id: int, customer_id: str, proposal: dict, flags: list[str], violations: list[str]) -> int:
        with self._con() as con:
            cur = con.execute(
                "INSERT INTO action_queue (run_id, customer_id, proposal_json, flags_json, violations_json, created_at) "
                "VALUES (?,?,?,?,?,?)",
                (run_id, customer_id, json.dumps(proposal), json.dumps(flags), json.dumps(violations), _now()))
            return cur.lastrowid

    def queue(self, status: str | None = "pending") -> list[dict]:
        with self._con() as con:
            rows = con.execute("SELECT * FROM action_queue" + (" WHERE status = ?" if status else "") +
                               " ORDER BY queue_id", (status,) if status else ()).fetchall()
        return [dict(r) for r in rows]

    def decide(self, queue_id: int, status: str, note: str = "") -> None:
        if status not in ("approved", "rejected"):
            raise ValueError("status must be 'approved' or 'rejected'")
        with self._con() as con:
            con.execute("UPDATE action_queue SET status=?, reviewer_note=?, decided_at=? WHERE queue_id=?",
                        (status, note, _now(), queue_id))

    def runs(self, config: str | None = None) -> list[dict]:
        with self._con() as con:
            rows = con.execute("SELECT * FROM agent_runs" + (" WHERE config = ?" if config else "") +
                               " ORDER BY run_id", (config,) if config else ()).fetchall()
        return [dict(r) for r in rows]
