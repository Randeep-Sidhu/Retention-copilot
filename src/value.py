"""Business case: expected-value targeting, policy comparison, profit curve, sensitivity analysis.

Decisions use PREDICTED probabilities only. Evaluation uses the REAL churn labels of held-out customers
combined with the ASSUMED treatment effects in `Economics` (no experiment exists, so no causal claims).
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from .config import Economics


def value_terms(mrr, econ: Economics):
    v_saved = mrr * econ.margin * econ.saved_lifetime_months      # margin kept if the customer stays
    d_cost = mrr * econ.discount_pct * econ.discount_months       # discount given to whoever accepts
    return v_saved, d_cost


def expected_value(p, mrr, econ: Economics):
    """EV of contacting a customer with churn probability p and monthly charge mrr."""
    v, d = value_terms(np.asarray(mrr, float), econ)
    p = np.asarray(p, float)
    return p * econ.save_rate * (v - d) - (1 - p) * econ.takeup_non_churners * d - econ.contact_cost


def realized_value(y, mrr, contact, econ: Economics) -> float:
    """Assumed-effect value of a contact list on customers whose true outcome is known."""
    v, d = value_terms(np.asarray(mrr, float), econ)
    y = np.asarray(y, float)
    per = y * econ.save_rate * (v - d) - (1 - y) * econ.takeup_non_churners * d - econ.contact_cost
    return float(per[np.asarray(contact, bool)].sum())


def _row(name, contact, y, mrr, econ):
    contact = np.asarray(contact, bool)
    n = len(contact)
    return {
        "policy": name,
        "contacted_pct": 100 * contact.mean(),
        "churners_reached_pct": 100 * (contact & (y == 1)).sum() / max(1, (y == 1).sum()),
        "precision_pct": 100 * (y[contact].mean() if contact.any() else 0.0),
        "net_value_per_1000": 1000 * realized_value(y, mrr, contact, econ) / n,
    }


def policy_comparison(p, y, mrr, econ: Economics, budget_frac: float = 0.20) -> pd.DataFrame:
    p, y, mrr = np.asarray(p), np.asarray(y), np.asarray(mrr, float)
    n = len(p)
    ev = expected_value(p, mrr, econ)
    top_k = np.zeros(n, bool)
    top_k[np.argsort(-p)[: int(round(budget_frac * n))]] = True
    rows = [
        _row("Contact nobody", np.zeros(n, bool), y, mrr, econ),
        _row("Contact everyone", np.ones(n, bool), y, mrr, econ),
        _row(f"Top {int(budget_frac * 100)}% by churn risk", top_k, y, mrr, econ),
        _row("Expected value > 0 (proposed)", ev > 0, y, mrr, econ),
    ]
    return pd.DataFrame(rows)


def profit_curve(p, y, mrr, econ: Economics) -> pd.DataFrame:
    """Net value per 1,000 customers as we contact customers in decreasing EV order."""
    p, y, mrr = np.asarray(p), np.asarray(y, float), np.asarray(mrr, float)
    order = np.argsort(-expected_value(p, mrr, econ))
    v, d = value_terms(mrr, econ)
    per = y * econ.save_rate * (v - d) - (1 - y) * econ.takeup_non_churners * d - econ.contact_cost
    cum = np.cumsum(per[order])
    n = len(p)
    return pd.DataFrame({"pct_contacted": 100 * np.arange(1, n + 1) / n, "net_value_per_1000": 1000 * cum / n})


def sensitivity_grid(p, y, mrr, base: Economics, save_rates, discounts) -> pd.DataFrame:
    """Re-run the EV>0 policy for a grid of save-rate x discount assumptions."""
    p, y, mrr = np.asarray(p), np.asarray(y), np.asarray(mrr, float)
    rows = []
    for s in save_rates:
        for d in discounts:
            econ = replace(base, save_rate=s, discount_pct=d)
            contact = expected_value(p, mrr, econ) > 0
            rows.append({"save_rate": s, "discount_pct": d,
                         "contacted_pct": 100 * contact.mean(),
                         "net_value_per_1000": 1000 * realized_value(y, mrr, contact, econ) / len(p)})
    return pd.DataFrame(rows)
