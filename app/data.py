"""Cached, read-only data access for the Streamlit app (the review queue is the only thing that writes)."""
from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st

from src.config import DB_PATH, REPORT_DIR, ROOT, RUNS_DB_PATH, TENURE_LABELS
from src.retriever import Retriever, load_corpus
from src.sql_tool import SqlTool
from src.store import RunStore

DEFAULTS = {"db": ("RC_DB", DB_PATH), "eval": ("RC_EVAL_DB", ROOT / "db" / "eval_runs.db"),
            "dev": ("RC_DEV_DB", ROOT / "db" / "dev_runs.db"), "runs": ("RC_RUNS_DB", RUNS_DB_PATH),
            "reports": ("RC_REPORTS", REPORT_DIR)}


def path(name: str) -> Path:
    """Locations can be overridden with environment variables (used by the tests)."""
    env, default = DEFAULTS[name]
    return Path(os.getenv(env, str(default)))


def _json(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError):
        return None


def _ro(db: str) -> sqlite3.Connection:
    return sqlite3.connect(f"{Path(db).resolve().as_uri()}?mode=ro", uri=True)


@st.cache_data(show_spinner=False)
def metrics(reports: str):
    return _json(Path(reports) / "metrics.json")


@st.cache_data(show_spinner=False)
def test_frame(db: str) -> pd.DataFrame:
    con = _ro(db)
    try:
        return pd.read_sql_query(
            "SELECT s.customer_id, s.p_churn, c.monthly_charges, l.churn_actual FROM scores s "
            "JOIN customers c USING (customer_id) JOIN labels l USING (customer_id) WHERE l.split = 'test'", con)
    finally:
        con.close()


@st.cache_data(show_spinner=False)
def tenure_table(db: str) -> pd.DataFrame:
    con = _ro(db)
    try:
        df = pd.read_sql_query("SELECT c.tenure_band AS band, COUNT(*) AS customers, AVG(l.churn_actual) AS churn_rate "
                               "FROM customers c JOIN labels l USING (customer_id) GROUP BY c.tenure_band", con)
    finally:
        con.close()
    order = {b: i for i, b in enumerate(TENURE_LABELS)}
    return df.sort_values("band", key=lambda s: s.map(order)).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def profile(db: str, customer_id: str):
    return SqlTool(db).profile(customer_id)


def run_sets() -> list[dict]:
    sets = []
    for label, key in (("Final evaluation (held-out customers)", "eval"), ("Development runs", "dev")):
        p = path(key)
        if p.exists():
            rows = RunStore(p).runs()
            if rows:
                sets.append({"label": label, "path": str(p), "models": sorted({r["model"] for r in rows})})
    return sets


@st.cache_data(show_spinner=False)
def runs(db_path: str, model: str) -> list[dict]:
    out = []
    for r in RunStore(db_path).runs():
        if r["model"] != model:
            continue
        for key in ("proposal", "first_violations", "violations", "retrieved_ids", "trace"):
            raw = r.pop(f"{key}_json", None)
            r[key] = json.loads(raw) if raw else None
        out.append(r)
    return out


@st.cache_data(show_spinner=False)
def model_reports(reports: str, stem: str) -> dict[str, dict]:
    out = {}
    for f in sorted(Path(reports).glob(f"{stem}_*.json")):
        m = _json(f)
        if m:
            out[f.stem[len(stem) + 1:]] = m
    return out


@st.cache_data(show_spinner=False)
def retrieval_report(reports: str):
    return _json(Path(reports) / "retrieval_eval.json")


@st.cache_resource(show_spinner=False)
def corpus() -> dict:
    return {c.id: c for c in load_corpus()}


@st.cache_resource(show_spinner=False)
def retriever(backend: str = "tfidf") -> Retriever:
    return Retriever(load_corpus(), backend=backend)
