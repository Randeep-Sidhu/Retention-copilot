"""Data loading, cleaning and the simulated policy-state fields."""
from __future__ import annotations

import urllib.request

import numpy as np
import pandas as pd

from .config import DATA_DIR, DATA_URL, RAW_CSV, RENAME, SEED, TENURE_BINS, TENURE_LABELS


def download_raw(force: bool = False):
    """Download the public CSV once (cached in data/, which is git-ignored)."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if force or not RAW_CSV.exists():
        print(f"Downloading dataset -> {RAW_CSV}")
        urllib.request.urlretrieve(DATA_URL, RAW_CSV)
    return RAW_CSV


def load_raw() -> pd.DataFrame:
    return pd.read_csv(download_raw())


def clean(raw: pd.DataFrame) -> pd.DataFrame:
    """snake_case columns, numeric total_charges, 0/1 target. Sorted by id for determinism."""
    df = raw.rename(columns=RENAME).copy()
    total = pd.to_numeric(df["total_charges"].astype(str).str.strip(), errors="coerce")
    blank = total.isna()
    # The blanks are customers with tenure 0 (not billed yet) -> 0.0 is exact, not an imputed guess.
    if not (df.loc[blank, "tenure_months"] == 0).all():
        raise ValueError("Unexpected blank total_charges for customers with tenure > 0")
    df["total_charges"] = total.fillna(0.0)
    df["senior_citizen"] = df["senior_citizen"].astype(int)
    df["churn"] = (df["churn"].astype(str).str.strip() == "Yes").astype(int)
    df["tenure_band"] = pd.cut(df["tenure_months"], bins=TENURE_BINS, labels=TENURE_LABELS).astype(str)
    return df.sort_values("customer_id").reset_index(drop=True)


def add_simulated_fields(df: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """Add marketing consent + recent-contact count (SIMULATED, seeded). Used only by policy rules."""
    rng = np.random.default_rng(seed)
    out = df.copy()
    out["marketing_opt_in"] = (rng.random(len(out)) < 0.88).astype(int)
    out["contacts_last_90d"] = np.minimum(rng.poisson(0.6, len(out)), 3).astype(int)
    return out
