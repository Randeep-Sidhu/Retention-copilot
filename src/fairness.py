"""Outcome audit by demographic group (attributes are NOT model inputs; this checks for proxy effects)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def group_report(y, p, contact, groups) -> pd.DataFrame:
    y, p, contact = np.asarray(y), np.asarray(p), np.asarray(contact, bool)
    groups = np.asarray(groups)
    rows = []
    for g in sorted(pd.unique(groups)):
        m = groups == g
        pos, neg = m & (y == 1), m & (y == 0)
        rows.append({
            "group": g, "n": int(m.sum()),
            "actual_churn_rate": float(y[m].mean()),
            "mean_predicted_risk": float(p[m].mean()),
            "contact_rate": float(contact[m].mean()),
            "churners_reached_tpr": float(contact[pos].mean()) if pos.any() else float("nan"),
            "non_churners_contacted_fpr": float(contact[neg].mean()) if neg.any() else float("nan"),
        })
    return pd.DataFrame(rows)


def gaps(report: pd.DataFrame) -> dict:
    sel = report["contact_rate"]
    return {
        "contact_rate_ratio_min_over_max": float(sel.min() / sel.max()) if sel.max() > 0 else float("nan"),
        "tpr_gap": float(report["churners_reached_tpr"].max() - report["churners_reached_tpr"].min()),
        "fpr_gap": float(report["non_churners_contacted_fpr"].max() - report["non_churners_contacted_fpr"].min()),
        "calibration_gap": float((report["mean_predicted_risk"] - report["actual_churn_rate"]).abs().max()),
    }
