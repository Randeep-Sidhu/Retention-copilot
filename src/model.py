"""Churn models, evaluation, paired bootstrap, and exact additive explanations for the logistic model."""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import SEED, TEST_SIZE


# ----------------------------------------------------------------------------- splitting / pipelines
def split(df: pd.DataFrame, test_size: float = TEST_SIZE, seed: int = SEED):
    train, test = train_test_split(df, test_size=test_size, stratify=df["churn"], random_state=seed)
    return train.copy(), test.copy()


def split_columns(X: pd.DataFrame):
    cat = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    num = [c for c in X.columns if c not in cat]
    return cat, num


def make_logreg(cat, num) -> Pipeline:
    pre = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat),
         ("num", StandardScaler(), num)],
        sparse_threshold=0.0, verbose_feature_names_out=False)
    return Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=3000))])


def make_hgb(cat, num) -> Pipeline:
    pre = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat)],
        remainder="passthrough", sparse_threshold=0.0)
    return Pipeline([("pre", pre), ("clf", HistGradientBoostingClassifier(random_state=SEED))])


LOGREG_GRID = {"clf__C": [0.05, 0.1, 0.5, 1.0, 5.0]}
HGB_GRID = {"clf__max_depth": [2, 3], "clf__learning_rate": [0.05], "clf__max_iter": [100, 200, 300]}


def tune(pipe: Pipeline, grid: dict, X, y):
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    gs = GridSearchCV(pipe, grid, cv=cv, scoring="roc_auc", n_jobs=1, refit=True)
    gs.fit(X, y)
    return gs.best_estimator_, float(gs.best_score_), gs.best_params_


# ----------------------------------------------------------------------------- metrics
def expected_calibration_error(y, p, n_bins: int = 10) -> float:
    y, p = np.asarray(y), np.asarray(p)
    idx = np.clip(np.digitize(p, np.linspace(0, 1, n_bins + 1)) - 1, 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(ece)


def top_decile_lift(y, p) -> float:
    y, p = np.asarray(y), np.asarray(p)
    k = max(1, int(round(0.1 * len(y))))
    top = np.argsort(-p)[:k]
    return float(y[top].mean() / y.mean())


def evaluate(y, p) -> dict:
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p)),
        "ece": expected_calibration_error(y, p),
        "top_decile_lift": top_decile_lift(y, p),
    }


def paired_bootstrap_diff(y, p_a, p_b, metric_fn, n_boot: int = 1000, seed: int = 0) -> dict:
    """95% percentile CI for metric(p_a) - metric(p_b), resampling the SAME rows for both models."""
    y, p_a, p_b = np.asarray(y), np.asarray(p_a), np.asarray(p_b)
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        diffs.append(metric_fn(y[i], p_a[i]) - metric_fn(y[i], p_b[i]))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"mean": float(np.mean(diffs)), "lo": float(lo), "hi": float(hi)}


def choose_model(delta_auc: dict) -> tuple[str, str]:
    """Prefer the interpretable model unless gradient boosting is significantly better."""
    if delta_auc["lo"] > 0:
        return "hist_gb", ("Gradient boosting is significantly better (95% CI of delta-AUC excludes 0); "
                           "it is used for probabilities and the logistic model supplies surrogate explanations.")
    return "logreg", ("The 95% CI of delta-AUC (gradient boosting minus logistic) includes 0: no significant gain, "
                      "so the interpretable logistic model is shipped.")


# ----------------------------------------------------------------------------- explanations
def feature_groups(pre: ColumnTransformer, cat, num) -> np.ndarray:
    """Original column name for every transformed column (cat block first, then numeric)."""
    ohe = pre.named_transformers_["cat"]
    groups = []
    for col, cats in zip(cat, ohe.categories_):
        groups += [col] * len(cats)
    return np.array(groups + list(num))


@dataclass
class Explainer:
    """Exact additive explanation of a fitted logistic pipeline, relative to the average training customer.

    logit(x) = base_logit + sum_f contribution_f(x)   (an identity, checked in tests)
    """
    pipe: Pipeline
    cat: list
    num: list
    z_mean: np.ndarray

    @classmethod
    def fit(cls, pipe: Pipeline, X_train: pd.DataFrame):
        cat, num = split_columns(X_train)
        z_mean = pipe.named_steps["pre"].transform(X_train).mean(axis=0)
        return cls(pipe, cat, num, z_mean)

    def contributions(self, X: pd.DataFrame):
        pre, clf = self.pipe.named_steps["pre"], self.pipe.named_steps["clf"]
        coef = clf.coef_[0]
        Z = pre.transform(X)
        per_col = (Z - self.z_mean) * coef
        groups = feature_groups(pre, self.cat, self.num)
        frame = pd.DataFrame(per_col).T.groupby(groups).sum().T
        frame.index = X.index
        base = float(clf.intercept_[0] + self.z_mean @ coef)
        return frame, base

    def drivers(self, X: pd.DataFrame, n_up: int = 3, n_down: int = 2):
        """Per customer: top risk-increasing and risk-decreasing factors as JSON strings."""
        contrib, _ = self.contributions(X)
        ups, downs = [], []
        for idx, row in contrib.iterrows():
            order = row.sort_values()
            def pack(feats):
                return json.dumps([
                    {"feature": f, "value": _plain(X.loc[idx, f]), "log_odds": round(float(row[f]), 3)}
                    for f in feats])
            up = [f for f in order.index[::-1][:n_up] if row[f] > 0]
            down = [f for f in order.index[:n_down] if row[f] < 0]
            ups.append(pack(up)); downs.append(pack(down))
        return ups, downs


def _plain(v):
    return v.item() if hasattr(v, "item") else v
