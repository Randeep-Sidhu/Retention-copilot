"""Fast, offline unit tests (no network, no LLM). Run: pytest -q"""
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from src import data as D
from src import fairness as F
from src import model as M
from src import value as V
from src.config import FEATURES, PROTECTED, RENAME, Economics


def toy_raw(n=300, seed=0) -> pd.DataFrame:
    """Synthetic frame with the ORIGINAL Telco column names and the blank-TotalCharges quirk."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "customerID": [f"C{i:04d}" for i in range(n)],
        "gender": rng.choice(["Male", "Female"], n),
        "SeniorCitizen": rng.integers(0, 2, n),
        "Partner": rng.choice(["Yes", "No"], n), "Dependents": rng.choice(["Yes", "No"], n),
        "tenure": rng.integers(1, 72, n),
        "PhoneService": rng.choice(["Yes", "No"], n),
        "MultipleLines": rng.choice(["Yes", "No", "No phone service"], n),
        "InternetService": rng.choice(["DSL", "Fiber optic", "No"], n),
        "OnlineSecurity": rng.choice(["Yes", "No", "No internet service"], n),
        "OnlineBackup": rng.choice(["Yes", "No", "No internet service"], n),
        "DeviceProtection": rng.choice(["Yes", "No", "No internet service"], n),
        "TechSupport": rng.choice(["Yes", "No", "No internet service"], n),
        "StreamingTV": rng.choice(["Yes", "No", "No internet service"], n),
        "StreamingMovies": rng.choice(["Yes", "No", "No internet service"], n),
        "Contract": rng.choice(["Month-to-month", "One year", "Two year"], n),
        "PaperlessBilling": rng.choice(["Yes", "No"], n),
        "PaymentMethod": rng.choice(["Electronic check", "Mailed check", "Credit card (automatic)"], n),
        "MonthlyCharges": rng.uniform(20, 110, n).round(2),
    })
    df["TotalCharges"] = (df["MonthlyCharges"] * df["tenure"]).round(2).astype(str)
    logit = -1.5 + 1.2 * (df["Contract"] == "Month-to-month") - 0.03 * df["tenure"] + 0.01 * df["MonthlyCharges"]
    df["Churn"] = np.where(rng.random(n) < 1 / (1 + np.exp(-logit)), "Yes", "No")
    df.loc[0, ["tenure", "TotalCharges"]] = [0, " "]          # the real dataset's blank-TotalCharges rows
    return df


@pytest.fixture(scope="module")
def cleaned():
    return D.add_simulated_fields(D.clean(toy_raw()))


# ------------------------------------------------------------------ data
def test_clean_fills_blank_total_charges_for_new_customers(cleaned):
    assert cleaned["total_charges"].dtype.kind == "f"
    assert cleaned.loc[cleaned["tenure_months"] == 0, "total_charges"].eq(0.0).all()
    assert set(cleaned["churn"].unique()) <= {0, 1}
    assert set(RENAME.values()) <= set(cleaned.columns)


def test_clean_rejects_blank_total_charges_for_billed_customers():
    raw = toy_raw()
    raw.loc[5, "TotalCharges"] = " "          # tenure > 0 -> data problem, must not be silently imputed
    with pytest.raises(ValueError):
        D.clean(raw)


def test_simulated_fields_are_deterministic_and_in_range(cleaned):
    again = D.add_simulated_fields(D.clean(toy_raw()))
    pd.testing.assert_frame_equal(cleaned, again)
    assert set(cleaned["marketing_opt_in"].unique()) <= {0, 1}
    assert cleaned["contacts_last_90d"].between(0, 3).all()


def test_protected_attributes_are_never_model_inputs():
    assert not set(PROTECTED) & set(FEATURES)


# ------------------------------------------------------------------ model
def test_additive_explanations_reconstruct_the_logit(cleaned):
    train, test = M.split(cleaned)
    cat, num = M.split_columns(train[FEATURES])
    pipe = M.make_logreg(cat, num).fit(train[FEATURES], train["churn"])
    expl = M.Explainer.fit(pipe, train[FEATURES])
    contrib, base = expl.contributions(test[FEATURES])
    np.testing.assert_allclose(base + contrib.sum(axis=1).to_numpy(),
                               pipe.decision_function(test[FEATURES]), rtol=1e-6, atol=1e-6)
    assert set(contrib.columns) == set(FEATURES)      # one contribution per ORIGINAL feature


def test_paired_bootstrap_of_identical_models_is_centred_on_zero(cleaned):
    y = cleaned["churn"].to_numpy()
    p = np.random.default_rng(1).random(len(y))
    d = M.paired_bootstrap_diff(y, p, p, roc_auc_score, n_boot=100)
    assert d["lo"] == d["hi"] == d["mean"] == 0.0


def test_choose_model_prefers_interpretable_unless_significantly_worse():
    assert M.choose_model({"lo": -0.01, "hi": 0.01})[0] == "logreg"
    assert M.choose_model({"lo": 0.002, "hi": 0.01})[0] == "hist_gb"


def test_calibration_error_zero_for_perfect_bins():
    y = np.array([0, 0, 1, 1]); p = np.array([0.0, 0.0, 1.0, 1.0])
    assert M.expected_calibration_error(y, p) == pytest.approx(0.0)


# ------------------------------------------------------------------ business case
def test_expected_value_matches_hand_calculation():
    econ = Economics(save_rate=0.25, takeup_non_churners=0.25, discount_pct=0.15, discount_months=6,
                     margin=0.35, saved_lifetime_months=12, contact_cost=3.0)
    mrr, p = 100.0, 0.4
    v, d = 100 * 0.35 * 12, 100 * 0.15 * 6                 # 420 and 90
    expected = 0.4 * 0.25 * (v - d) - 0.6 * 0.25 * d - 3.0
    assert V.expected_value(p, mrr, econ) == pytest.approx(expected)


def test_ev_increases_with_churn_probability():
    econ = Economics()
    ev = V.expected_value(np.linspace(0, 1, 20), np.full(20, 70.0), econ)
    assert np.all(np.diff(ev) > 0)


def test_contact_nobody_is_worth_zero_and_policy_table_is_consistent():
    rng = np.random.default_rng(0)
    y = (rng.random(500) < 0.3).astype(int)
    p = np.clip(0.3 + 0.4 * (y - 0.3) + rng.normal(0, 0.1, 500), 0.01, 0.99)
    mrr = rng.uniform(30, 100, 500)
    pol = V.policy_comparison(p, y, mrr, Economics()).set_index("policy")
    assert pol.loc["Contact nobody", "net_value_per_1000"] == 0
    assert pol.loc["Contact everyone", "contacted_pct"] == 100
    curve = V.profit_curve(p, y, mrr, Economics())
    assert curve["net_value_per_1000"].iloc[-1] == pytest.approx(pol.loc["Contact everyone", "net_value_per_1000"])


# ------------------------------------------------------------------ fairness
def test_group_report_rates():
    y = np.array([1, 1, 0, 0, 1, 0]); p = np.array([.9, .8, .2, .1, .6, .7])
    contact = np.array([1, 0, 0, 0, 1, 1], bool); groups = np.array(["a", "a", "a", "b", "b", "b"])
    rep = F.group_report(y, p, contact, groups).set_index("group")
    assert rep.loc["a", "contact_rate"] == pytest.approx(1 / 3)
    assert rep.loc["a", "churners_reached_tpr"] == pytest.approx(0.5)
    assert rep.loc["b", "non_churners_contacted_fpr"] == pytest.approx(0.5)
    assert F.gaps(rep)["tpr_gap"] == pytest.approx(0.5)
