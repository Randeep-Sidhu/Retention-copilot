"""Read-only SQL tool for the agent.

Safety is enforced by the database, not by the prompt: the connection is opened read-only, and a SQLite
authorizer allows nothing except SELECT on the view `v_customer_profile` (which has no churn label and no
demographic attributes). Direct access to any table, PRAGMA, ATTACH, writes or multiple statements is denied.
"""
from __future__ import annotations

import re
import sqlite3
from pathlib import Path

from .config import DB_PATH

ALLOWED_VIEW = "v_customer_profile"
UNDERLYING_TABLES = {"customers", "scores"}          # readable ONLY when reached through the view
DENIED_FUNCTIONS = {"load_extension", "readfile", "writefile", "edit"}


class SqlNotAllowed(Exception):
    pass


def _authorizer(action, arg1, arg2, db_name, source):
    if action == sqlite3.SQLITE_SELECT or action == sqlite3.SQLITE_RECURSIVE:
        return sqlite3.SQLITE_OK
    if action == sqlite3.SQLITE_FUNCTION:
        return sqlite3.SQLITE_DENY if (arg2 or "").lower() in DENIED_FUNCTIONS else sqlite3.SQLITE_OK
    if action == sqlite3.SQLITE_READ:
        if arg1 == ALLOWED_VIEW:
            return sqlite3.SQLITE_OK
        if arg1 in UNDERLYING_TABLES and source == ALLOWED_VIEW:
            return sqlite3.SQLITE_OK
        return sqlite3.SQLITE_DENY
    return sqlite3.SQLITE_DENY


class SqlTool:
    def __init__(self, db_path: Path | str = DB_PATH, max_rows: int = 50):
        self.db_path, self.max_rows = Path(db_path), max_rows

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(f"{self.db_path.resolve().as_uri()}?mode=ro", uri=True)
        con.row_factory = sqlite3.Row
        con.set_authorizer(_authorizer)
        return con

    def query(self, sql: str, params: tuple | dict = ()) -> list[dict]:
        text = sql.strip().rstrip(";").strip()
        if not re.match(r"(?is)^(select|with)\b", text) or "--" in text or "/*" in text:
            raise SqlNotAllowed("only a single plain SELECT statement is allowed")
        con = self._connect()
        try:
            try:
                cur = con.execute(text, params)
            except sqlite3.DatabaseError as exc:                # authorizer denials, multi-statement, syntax
                raise SqlNotAllowed(f"query rejected: {exc}") from exc
            return [dict(r) for r in cur.fetchmany(self.max_rows)]
        finally:
            con.close()

    def profile(self, customer_id: str) -> dict | None:
        rows = self.query(f"SELECT * FROM {ALLOWED_VIEW} WHERE customer_id = ?", (customer_id,))
        return rows[0] if rows else None
