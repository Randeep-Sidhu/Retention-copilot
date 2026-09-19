"""Phase 1 pipeline: data -> models -> statistical comparison -> business case -> fairness audit -> artifacts.

Run:  python -m src.phase1
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from dataclasses import asdict

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402

from . import data as D  # noqa: E402
from . import fairness as F  # noqa: E402
from . import model as M  # noqa: E402
from . import value as V  # noqa: E402
from .config import (ARTIFACT_DIR, DB_EXTRA, DB_PATH, DROPPED_COLLINEAR, FEATURES, FIG_DIR,  # noqa: E402
                     PROTECTED, REPORT_DIR, SEED, TENURE_LABELS, Economics)

ECON = Economics()
SAVE_RATES = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40]
DISCOUNTS = [0.05, 0.10, 0.15, 0.20, 0.25]


# ----------------------------------------------------------------------------- database
def write_db(df, p_all, ups, downs, split_flag, model_name):
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    if DB_PATH.exists():
        DB_PATH.unlink()
    con = sqlite3.connect(DB_PATH)
    cust_cols = ["customer_id"] + FEATURES + DB_EXTRA + ["marketing_opt_in", "contacts_last_90d"]
    df[cust_cols].to_sql("customers", con, index=False)
    df[["customer_id"] + PROTECTED].to_sql("audit_attributes", con, index=False)
    pd.DataFrame({"customer_id": df["customer_id"], "churn_actual": df["churn"],
                  "split": split_flag}).to_sql("labels", con, index=False)
    pd.DataFrame({"customer_id": df["customer_id"], "p_churn": np.round(p_all, 4),
                  "drivers_json": ups, "protective_json": downs, "model": model_name}
                 ).to_sql("scores", con, index=False)
    con.execute("CREATE UNIQUE INDEX ix_customers ON customers(customer_id)")
    con.execute("CREATE UNIQUE INDEX ix_scores ON scores(customer_id)")
    # The ONLY object the agent will be allowed to query: no label, no demographics.
    con.execute("""CREATE VIEW v_customer_profile AS
                   SELECT c.*, s.p_churn, s.drivers_json, s.protective_json
                   FROM customers c JOIN scores s USING (customer_id)""")
    con.commit()
    con.close()


# ----------------------------------------------------------------------------- figures
def fig_calibration(y, probs: dict):
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1, label="perfect")
    for name, p in probs.items():
        frac, mean = calibration_curve(y, p, n_bins=10, strategy="quantile")
        ax.plot(mean, frac, marker="o", lw=1.5, label=name)
    ax.set_xlabel("Predicted churn probability"); ax.set_ylabel("Observed churn rate")
    ax.set_title("Calibration on held-out customers"); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(FIG_DIR / "calibration.png", dpi=150); plt.close(fig)


def fig_profit(curve, policies):
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.plot(curve["pct_contacted"], curve["net_value_per_1000"], color="#1f4e79", lw=2,
            label="Contact in order of expected value")
    ax.axhline(0, color="grey", lw=0.8)
    for _, r in policies.iterrows():
        if r["policy"].startswith(("Expected", "Top")):
            ax.scatter(r["contacted_pct"], r["net_value_per_1000"], zorder=3, s=45,
                       label=f'{r["policy"]}: ${r["net_value_per_1000"]:,.0f}')
    ax.set_xlabel("% of customers contacted"); ax.set_ylabel("Net value per 1,000 customers ($)")
    ax.set_title("Who to contact: value vs. reach (assumed economics)"); ax.legend(frameon=False, fontsize=8, loc="lower center")
    fig.tight_layout(); fig.savefig(FIG_DIR / "profit_curve.png", dpi=150); plt.close(fig)


def fig_sensitivity(grid):
    piv = grid.pivot(index="save_rate", columns="discount_pct", values="net_value_per_1000")
    con = grid.pivot(index="save_rate", columns="discount_pct", values="contacted_pct")
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    im = ax.imshow(piv.values, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels([f"{int(c * 100)}%" for c in piv.columns])
    ax.set_yticks(range(len(piv.index))); ax.set_yticklabels([f"{int(r * 100)}%" for r in piv.index])
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            ax.text(j, i, f"${piv.values[i, j]:,.0f}\n{con.values[i, j]:.0f}% contacted", ha="center",
                    va="center", fontsize=7,
                    color="white" if piv.values[i, j] > 0.6 * piv.values.max() else "black")
    ax.set_xlabel("Discount depth"); ax.set_ylabel("Assumed save rate")
    ax.set_title("Net value per 1,000 customers vs. assumptions")
    fig.colorbar(im, ax=ax, shrink=0.8); fig.tight_layout()
    fig.savefig(FIG_DIR / "sensitivity.png", dpi=150); plt.close(fig)


def fig_drivers(mean_abs):
    top = mean_abs.sort_values().tail(10)
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    ax.barh(top.index, top.values, color="#1f4e79")
    ax.set_xlabel("Mean |contribution| to churn log-odds"); ax.set_title("What drives predicted churn")
    fig.tight_layout(); fig.savefig(FIG_DIR / "drivers.png", dpi=150); plt.close(fig)


# ----------------------------------------------------------------------------- markdown helpers
def fmt(v, kind):
    if isinstance(v, str):
        return v
    return {"p": f"{v:.3f}", "pct": f"{v:.1f}%", "usd": f"${v:,.0f}", "int": f"{int(v)}",
            "r": f"{v:.2f}"}.get(kind, f"{v:.3f}")


def md_table(df: pd.DataFrame, kinds: dict | None = None) -> str:
    kinds = kinds or {}
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(fmt(r[c], kinds.get(c, "p")) for c in cols) + " |")
    return "\n".join(lines)


def ci_text(d):
    return f"{d['mean']:+.4f} (95% CI {d['lo']:+.4f} to {d['hi']:+.4f})"


# ----------------------------------------------------------------------------- main
def main(n_boot: int = 1000):
    for d in (ARTIFACT_DIR, REPORT_DIR, FIG_DIR):
        d.mkdir(parents=True, exist_ok=True)

    print("1/6 Loading and cleaning data ...")
    df = D.add_simulated_fields(D.clean(D.load_raw()))
    train, test = M.split(df)
    Xtr, ytr, Xte, yte = train[FEATURES], train["churn"], test[FEATURES], test["churn"]
    cat, num = M.split_columns(Xtr)
    # Fair benchmark: gradient boosting gets its natural inputs (numeric tenure and total_charges, no banding).
    GB_FEATURES = [f for f in FEATURES if f != "tenure_band"] + ["tenure_months", DROPPED_COLLINEAR]
    cat_g, num_g = M.split_columns(train[GB_FEATURES])
    print(f"    {len(df)} customers | churn rate {df['churn'].mean():.1%} | train {len(train)} / test {len(test)}")

    print("2/6 Training and tuning models (5-fold CV on train) ...")
    lr, lr_cv, lr_params = M.tune(M.make_logreg(cat, num), M.LOGREG_GRID, Xtr, ytr)
    gb, gb_cv, gb_params = M.tune(M.make_hgb(cat_g, num_g), M.HGB_GRID, train[GB_FEATURES], ytr)
    p_lr, p_gb = lr.predict_proba(Xte)[:, 1], gb.predict_proba(test[GB_FEATURES])[:, 1]
    m_lr, m_gb = M.evaluate(yte, p_lr), M.evaluate(yte, p_gb)

    print(f"3/6 Paired bootstrap ({n_boot} resamples) ...")
    d_auc = M.paired_bootstrap_diff(yte, p_gb, p_lr, roc_auc_score, n_boot, SEED)
    d_pr = M.paired_bootstrap_diff(yte, p_gb, p_lr, average_precision_score, n_boot, SEED)
    choice, why = M.choose_model(d_auc)
    selected = lr if choice == "logreg" else gb

    # Ablations (each is a paired bootstrap on the same test rows; C is held at the tuned value)
    C = lr_params["clf__C"]

    def lr_variant(cols):
        c, n = M.split_columns(train[cols])
        m = M.make_logreg(c, n).set_params(clf__C=C).fit(train[cols], ytr)
        return m.predict_proba(test[cols])[:, 1]

    demo_cols = FEATURES + PROTECTED                                   # + gender, age group, partner, dependents
    tc_cols = FEATURES + [DROPPED_COLLINEAR]                           # + total_charges
    numeric_tenure_cols = [f for f in FEATURES if f != "tenure_band"] + ["tenure_months"]
    d_demo = M.paired_bootstrap_diff(yte, lr_variant(demo_cols), p_lr, roc_auc_score, n_boot, SEED)
    d_tc = M.paired_bootstrap_diff(yte, lr_variant(tc_cols), p_lr, roc_auc_score, n_boot, SEED)
    p_lr_numeric = lr_variant(numeric_tenure_cols)
    d_band = M.paired_bootstrap_diff(yte, p_lr, p_lr_numeric, roc_auc_score, n_boot, SEED)          # banded - numeric (LR)
    d_gb_vs_numeric = M.paired_bootstrap_diff(yte, p_gb, p_lr_numeric, roc_auc_score, n_boot, SEED)  # GB - LR(numeric tenure)
    corr_tc = float(np.corrcoef(df[DROPPED_COLLINEAR], df["tenure_months"] * df["monthly_charges"])[0, 1])
    band_tbl = (df.groupby("tenure_band")["churn"].agg(customers="size", churn_rate="mean")
                .reindex(TENURE_LABELS).reset_index().rename(columns={"tenure_band": "tenure (months)", "churn_rate": "churn rate"}))

    print("4/6 Scoring, explanations and database ...")
    expl = M.Explainer.fit(lr, Xtr)
    used_features = FEATURES if choice == "logreg" else GB_FEATURES
    p_all = selected.predict_proba(df[used_features])[:, 1]
    ups, downs = expl.drivers(df[FEATURES])
    split_flag = np.where(df["customer_id"].isin(test["customer_id"]), "test", "train")
    write_db(df, p_all, ups, downs, split_flag, choice)
    joblib.dump({"selected": choice, "pipeline": selected, "logreg": lr, "explainer": expl,
                 "features": used_features, "logreg_features": FEATURES, "economics": asdict(ECON)}, ARTIFACT_DIR / "model.joblib")

    print("5/6 Business case and fairness audit ...")
    p_sel = p_lr if choice == "logreg" else p_gb
    y, mrr = yte.to_numpy(), test["monthly_charges"].to_numpy()
    policies = V.policy_comparison(p_sel, y, mrr, ECON)
    curve = V.profit_curve(p_sel, y, mrr, ECON)
    grid = V.sensitivity_grid(p_sel, y, mrr, ECON, SAVE_RATES, DISCOUNTS)
    contact = V.expected_value(p_sel, mrr, ECON) > 0
    audits = {}
    for attr in PROTECTED:
        rep = F.group_report(y, p_sel, contact, test[attr].to_numpy())
        audits[attr] = (rep, F.gaps(rep))

    print("6/6 Figures and reports ...")
    fig_calibration(yte, {"Logistic": p_lr, "Gradient boosting": p_gb})
    fig_profit(curve, policies)
    fig_sensitivity(grid)
    contrib, _ = expl.contributions(Xte)
    fig_drivers(contrib.abs().mean())

    proposed = policies[policies["policy"].str.startswith("Expected")].iloc[0]
    everyone = policies[policies["policy"] == "Contact everyone"].iloc[0]
    metrics = {
        "n_customers": int(len(df)), "churn_rate": float(df["churn"].mean()),
        "n_train": int(len(train)), "n_test": int(len(test)), "selected_model": choice,
        "logreg": {**m_lr, "cv_roc_auc": lr_cv, "params": lr_params},
        "hist_gb": {**m_gb, "cv_roc_auc": gb_cv, "params": gb_params},
        "delta_auc_gb_minus_lr": d_auc, "delta_prauc_gb_minus_lr": d_pr,
        "ablations": {"add_demographics": d_demo, "add_total_charges": d_tc, "banded_minus_numeric_tenure_lr": d_band,
                      "gb_minus_lr_numeric_tenure": d_gb_vs_numeric, "corr_total_charges_vs_tenure_x_monthly": corr_tc},
        "economics": asdict(ECON),
        "policies": policies.round(3).to_dict(orient="records"),
        "sensitivity_net_value_range_per_1000": [float(grid["net_value_per_1000"].min()),
                                                 float(grid["net_value_per_1000"].max())],
        "sensitivity_contacted_pct_range": [float(grid["contacted_pct"].min()),
                                            float(grid["contacted_pct"].max())],
        "fairness_gaps": {k: v[1] for k, v in audits.items()},
    }
    (REPORT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (REPORT_DIR / "model_card.md").write_text(
        model_card(metrics, m_lr, m_gb, d_auc, d_pr, why, policies, grid, audits, band_tbl), encoding="utf-8")

    print("\n=== PHASE 1 SUMMARY (paste this back) ===")
    print(f"Customers {len(df)} | churn {df['churn'].mean():.1%} | test n={len(test)}")
    print(f"Logistic  : ROC-AUC {m_lr['roc_auc']:.3f} | PR-AUC {m_lr['pr_auc']:.3f} | Brier {m_lr['brier']:.3f} | ECE {m_lr['ece']:.3f}")
    print(f"Grad.boost: ROC-AUC {m_gb['roc_auc']:.3f} | PR-AUC {m_gb['pr_auc']:.3f} | Brier {m_gb['brier']:.3f} | ECE {m_gb['ece']:.3f}")
    print(f"delta-AUC (GB - LR): {ci_text(d_auc)}")
    print(f"Selected model: {choice}. {why}")
    print(f"Tenure bands vs numeric tenure (logistic): {ci_text(d_band)}; GB vs numeric-tenure logistic: {ci_text(d_gb_vs_numeric)}")
    print(f"Adding demographics changes ROC-AUC by {ci_text(d_demo)}; adding total_charges (corr {corr_tc:.4f}) by {ci_text(d_tc)}")
    print("Policy comparison (per 1,000 customers, assumed economics):")
    for _, r in policies.iterrows():
        print(f"  {r['policy']:<34} contacted {r['contacted_pct']:5.1f}% | net ${r['net_value_per_1000']:>9,.0f}")
    print(f"Sensitivity: net value per 1,000 ranges ${metrics['sensitivity_net_value_range_per_1000'][0]:,.0f} "
          f"to ${metrics['sensitivity_net_value_range_per_1000'][1]:,.0f}; contact rate "
          f"{metrics['sensitivity_contacted_pct_range'][0]:.0f}%-{metrics['sensitivity_contacted_pct_range'][1]:.0f}%")
    for attr in ("senior_citizen", "gender"):
        g = audits[attr][1]
        print(f"Audit [{attr}]: contact-rate ratio {g['contact_rate_ratio_min_over_max']:.2f} | TPR gap {g['tpr_gap']:.3f} | calibration gap {g['calibration_gap']:.3f}")
    print(f"Database: {DB_PATH} | Model: {ARTIFACT_DIR / 'model.joblib'} | Reports: {REPORT_DIR}")
    print(f"(everyone-contacted baseline: ${everyone['net_value_per_1000']:,.0f} per 1,000; proposed: ${proposed['net_value_per_1000']:,.0f})")


def model_card(metrics, m_lr, m_gb, d_auc, d_pr, why, policies, grid, audits, band_tbl) -> str:
    econ = metrics["economics"]
    ab = metrics["ablations"]
    d_band, d_gbn, d_demo, d_tc, corr_tc = (ab["banded_minus_numeric_tenure_lr"], ab["gb_minus_lr_numeric_tenure"],
                                            ab["add_demographics"], ab["add_total_charges"],
                                            ab["corr_total_charges_vs_tenure_x_monthly"])
    comp = pd.DataFrame([
        {"model": "Logistic regression (banded tenure, no total charges)", "CV ROC-AUC": metrics["logreg"]["cv_roc_auc"], "test ROC-AUC": m_lr["roc_auc"],
         "PR-AUC": m_lr["pr_auc"], "Brier": m_lr["brier"], "ECE": m_lr["ece"], "top-decile lift": m_lr["top_decile_lift"]},
        {"model": "Gradient boosting (numeric tenure + total charges)", "CV ROC-AUC": metrics["hist_gb"]["cv_roc_auc"], "test ROC-AUC": m_gb["roc_auc"],
         "PR-AUC": m_gb["pr_auc"], "Brier": m_gb["brier"], "ECE": m_gb["ece"], "top-decile lift": m_gb["top_decile_lift"]},
    ])
    econ_tbl = pd.DataFrame([
        {"assumption": "Save rate (would-be churner stays after an offer)", "value": f"{econ['save_rate']:.0%}"},
        {"assumption": "Take-up of the discount by customers who would have stayed anyway", "value": f"{econ['takeup_non_churners']:.0%}"},
        {"assumption": "Standard offer", "value": f"{econ['discount_pct']:.0%} off for {econ['discount_months']} months"},
        {"assumption": "Contribution margin on monthly charges", "value": f"{econ['margin']:.0%}"},
        {"assumption": "Extra lifetime months if saved", "value": str(econ["saved_lifetime_months"])},
        {"assumption": "Cost per contact", "value": f"${econ['contact_cost']:.2f}"},
    ])
    pol = policies.rename(columns={"contacted_pct": "contacted", "churners_reached_pct": "churners reached",
                                   "precision_pct": "precision", "net_value_per_1000": "net value / 1,000"})
    kinds = {"contacted": "pct", "churners reached": "pct", "precision": "pct", "net value / 1,000": "usd"}
    parts = [
        "# Model card: churn risk model\n",
        "*Generated by `python -m src.phase1`. Numbers below come from the held-out test set "
        f"(n={metrics['n_test']}) unless stated.*\n",
        "## Intended use and limits\n"
        "Ranks customers by churn risk to prioritise retention outreach that a human reviews before it is sent. "
        "Data is a public sample dataset, not a real operator's. There is no offer/no-offer experiment in the data, "
        "so the business-value numbers rest on **stated assumptions**, not causal estimates.\n",
        "## Data\n"
        f"IBM's public Telco churn sample: {metrics['n_customers']:,} customers, {metrics['churn_rate']:.1%} churned. "
        "Stratified 75/25 train/test split, seed 42. 11 blank `total_charges` values belong to customers with tenure 0 "
        "and are set to 0. Two fields, `marketing_opt_in` and `contacts_last_90d`, are **simulated** (seeded) so that "
        "compliance rules can be tested; they are never model inputs.\n",
        "**Excluded from the model and from the agent-visible data:** gender, senior citizen, partner, dependents. "
        "They are kept in a separate audit table only.\n",
        "## Model comparison\n" + md_table(comp),
        f"\nPaired bootstrap (1,000 resamples of the same test rows), gradient boosting minus logistic: "
        f"ROC-AUC {ci_text(d_auc)}; PR-AUC {ci_text(d_pr)}.\n\n**Decision:** {why}\n",
        "Calibration is checked with Brier score, expected calibration error (ECE) and the reliability curve "
        "(`reports/figures/calibration.png`); the expected-value calculation needs probabilities that mean what they say.\n",
        "## Feature decisions (each checked with a paired bootstrap on the test rows)\n",
        "**Tenure is banded.** Churn falls steeply and non-linearly with tenure:\n\n"
        + md_table(band_tbl.assign(**{"churn rate": band_tbl["churn rate"] * 100}), {"customers": "int", "churn rate": "pct"})
        + f"\n\nWith tenure as a plain number the logistic model trailed gradient boosting significantly "
        f"(ROC-AUC {ci_text(d_gbn)}). Banding tenure raised the logistic model's ROC-AUC by {ci_text(d_band)} "
        "(small and, taken alone, within noise). After banding, the gap to gradient boosting is no longer "
        "distinguishable from zero (comparison above), and bands are easier to explain to a client. "
        "The banding decision came from the churn-by-tenure table, not from the AUC.\n",
        "**`total_charges` is dropped.** It is almost a deterministic function of tenure x monthly charges "
        f"(correlation {corr_tc:.4f}) and made per-customer explanations contradictory. Adding it back changes "
        f"ROC-AUC by {ci_text(d_tc)}.\n",
        "**Demographics are excluded.** Adding gender, age group, partner and dependents changes ROC-AUC by "
        f"{ci_text(d_demo)}: excluding them costs essentially nothing in accuracy.\n",
        "## Business case (assumed economics)\n" + md_table(econ_tbl),
        "\nDecision rule: contact a customer when expected value > 0, where "
        "`EV = p*s*(V-D) - (1-p)*a*D - c` (p churn probability, s save rate, a take-up by non-churners, "
        "V margin kept if saved, D discount cost, c contact cost). Evaluated on real held-out outcomes combined with the assumed effects.\n",
        md_table(pol, kinds),
        f"\nAcross the sensitivity grid (save rate {SAVE_RATES[0]:.0%}-{SAVE_RATES[-1]:.0%}, discount "
        f"{DISCOUNTS[0]:.0%}-{DISCOUNTS[-1]:.0%}) net value per 1,000 customers ranges from "
        f"${grid['net_value_per_1000'].min():,.0f} to ${grid['net_value_per_1000'].max():,.0f} and the share contacted from "
        f"{grid['contacted_pct'].min():.0f}% to {grid['contacted_pct'].max():.0f}%. "
        "Conclusions should be read as 'under these assumptions', which is why the grid is shown "
        "(`reports/figures/sensitivity.png`).\n",
        "## Fairness audit (outcomes by group; attributes are not model inputs)\n",
    ]
    for attr, (rep, g) in audits.items():
        r = rep.copy()
        r["group"] = r["group"].astype(str)
        r = r.rename(columns={"actual_churn_rate": "actual churn", "mean_predicted_risk": "mean predicted",
                              "contact_rate": "contact rate", "churners_reached_tpr": "churners reached (TPR)",
                              "non_churners_contacted_fpr": "non-churners contacted (FPR)"})
        parts.append(f"**{attr}**\n\n" + md_table(r, {"n": "int"}) +
                     f"\n\ncontact-rate ratio (min/max) {g['contact_rate_ratio_min_over_max']:.2f}; "
                     f"TPR gap {g['tpr_gap']:.3f}; FPR gap {g['fpr_gap']:.3f}; calibration gap {g['calibration_gap']:.3f}\n")
    parts.append(
        "Differences in contact rate partly reflect real differences in churn rates between groups. This audit reports "
        "them and does not claim the outcome is 'fair'; small groups have wide uncertainty.\n")
    parts.append("## Limitations\n"
                 "- No treatment data: save rate and take-up are assumptions.\n"
                 "- One public dataset, one time snapshot; no drift or seasonality analysis.\n"
                 "- Scores for training customers are in-sample; evaluation and agent scenarios use the held-out test customers.\n")
    return "\n".join(parts)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=1000, help="bootstrap resamples")
    main(ap.parse_args().boot)
