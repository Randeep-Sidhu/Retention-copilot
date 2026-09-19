"""Evaluation tooling (NOT an agent tool): picks held-out customers directly from the analytics database."""
from __future__ import annotations

import random
import sqlite3

from .config import DB_PATH, Economics
from .policy_engine import preferred_offer
from .policy_spec import HIGH_RISK_THRESHOLD, HIGH_VALUE_MONTHLY, MAX_CONTACTS_90D, NEW_CUSTOMER_MAX_TENURE
from .sql_tool import SqlTool
from .value import expected_value


def gate_passing_test_ids(db_path=DB_PATH, econ: Economics | None = None) -> list[str]:
    """Held-out (test-split) customers whose expected value is positive, i.e. the ones the agent would act on."""
    econ = econ or Economics()
    con = sqlite3.connect(db_path)
    rows = con.execute("SELECT s.customer_id, s.p_churn, c.monthly_charges FROM scores s "
                       "JOIN customers c USING (customer_id) JOIN labels l USING (customer_id) "
                       "WHERE l.split = 'test' ORDER BY s.customer_id").fetchall()
    con.close()
    return [cid for cid, p, mrr in rows if float(expected_value(p, mrr, econ)) > 0]


def sample_test_customers(n: int = 5, seed: int = 7, db_path=DB_PATH, econ: Economics | None = None) -> list[str]:
    ids = gate_passing_test_ids(db_path, econ)
    return random.Random(seed).sample(ids, min(n, len(ids)))


def tags(p: dict) -> set[str]:
    t = set()
    if not int(p["marketing_opt_in"]): t.add("opt_out")
    if int(p["contacts_last_90d"]) >= MAX_CONTACTS_90D: t.add("contact_limit")
    if int(p["tenure_months"]) <= NEW_CUSTOMER_MAX_TENURE: t.add("new_customer")
    if p["contract"] == "Month-to-month" and float(p["p_churn"]) >= HIGH_RISK_THRESHOLD: t.add("high_risk_m2m")
    if float(p["monthly_charges"]) > HIGH_VALUE_MONTHLY: t.add("high_value")
    if p["contract"] == "Two year": t.add("two_year")
    if p["contract"] == "One year": t.add("one_year")
    if p["internet_service"] == "No": t.add("phone_only")
    if p["contract"] == "Month-to-month" and p["internet_service"] == "DSL": t.add("m2m_dsl")
    if preferred_offer(p) == "LOYALTY_DISCOUNT": t.add("loyalty_preferred")     # exercises the numeric discount rules
    return t


QUOTAS = {"contact_limit": 4, "loyalty_preferred": 8, "opt_out": 6, "new_customer": 5, "high_risk_m2m": 6,
          "high_value": 5, "one_year": 4, "phone_only": 4, "m2m_dsl": 3}          # filled rarest-first; rest is random


def build_scenarios(n: int = 48, seed: int = 11, db_path=DB_PATH) -> list[str]:
    """Stratified, seeded scenario set: quotas for edge cases (no consent, contact limit, new, high risk, high
    value, contract types), then random fill. Deterministic for a given database."""
    ids = gate_passing_test_ids(db_path)
    sql = SqlTool(db_path)
    profiles = {cid: sql.profile(cid) for cid in ids}
    rng = random.Random(seed)
    pool = ids[:]
    rng.shuffle(pool)
    chosen: list[str] = []
    for tag, quota in QUOTAS.items():
        chosen += [c for c in pool if c not in chosen and tag in tags(profiles[c])][:quota]
    chosen += [c for c in pool if c not in chosen][: max(0, n - len(chosen))]
    return chosen[:n]


def scenario_mix(ids: list[str], db_path=DB_PATH) -> dict[str, int]:
    sql = SqlTool(db_path)
    mix: dict[str, int] = {}
    for cid in ids:
        for t in tags(sql.profile(cid)):
            mix[t] = mix.get(t, 0) + 1
    return dict(sorted(mix.items()))
