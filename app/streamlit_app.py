"""Retention Copilot review app.   Run:  streamlit run app/streamlit_app.py

Works without a language model: every agent run shown here was recorded during the evaluation and is replayed.
"""
from __future__ import annotations

import json
import re
import sys
import urllib.request
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import altair as alt  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from app import data, ui  # noqa: E402
from src.config import Economics  # noqa: E402
from src.policy_engine import review_flags  # noqa: E402
from src.schemas import Proposal  # noqa: E402
from src.store import RunStore  # noqa: E402
from src.value import policy_comparison, profit_curve, sensitivity_grid  # noqa: E402

REPO_URL = "https://github.com/Randeep-Sidhu/Retention-copilot"
CONFIG_ORDER = ["none", "none+verify", "full", "full+verify", "rag", "rag+verify"]
CONFIG_HELP = {"none": "no policy text", "none+verify": "no policy text, verifier loop", "full": "whole policy in the prompt",
               "full+verify": "whole policy, verifier loop", "rag": "retrieved policy sections",
               "rag+verify": "retrieved sections + verifier loop"}
MODEL_NAMES = {"granite4-micro": "Granite 4 Micro (3B)", "granite4:micro": "Granite 4 Micro (3B)",
               "granite4-small-h": "Granite 4 H-Small (32B MoE)", "granite4:small-h": "Granite 4 H-Small (32B MoE)"}
BRONZE, SANDBAR, ESPRESSO, TAUPE, TERRACOTTA = "#8C6A3F", "#D9C8AE", "#2A2420", "#7A6E63", "#B4574A"

st.set_page_config(page_title="Retention Copilot", page_icon="📞", layout="wide")
ui.inject()


def model_name(tag: str) -> str:
    return MODEL_NAMES.get(tag, tag)


def styled(chart, height: int = 300):
    return (chart.properties(height=height)
            .configure_view(strokeWidth=0)
            .configure_axis(labelColor=TAUPE, titleColor=TAUPE, gridColor="#EBE4D8", domainColor="#D8CDBD", tickColor="#D8CDBD",
                            labelFontSize=12, titleFontSize=12, titleFontWeight="normal")
            .configure_legend(labelColor=TAUPE, titleColor=TAUPE, labelFontSize=12, symbolType="square"))


def rate_text(d: dict) -> str:
    return f"{d['k']}/{d['n']} ({100 * d['rate']:.0f}%)"


def rule_ids(violations) -> list[str]:
    return [m.group(1) for v in (violations or []) if (m := re.match(r"^\[(\w+)\]", v))]


def trace_detail(e: dict) -> tuple[str, str]:
    n = e["node"]
    if n == "load_customer":
        return (e.get("error") or f"{e.get('contract')}, {e.get('internet_service')}, tenure {ui.plural(e.get('tenure_months', 0), 'month')}"), ""
    if n == "gate":
        return f"expected value ${e['expected_value']:.2f} per contact, {'proceed' if e['proceed'] else 'skip'}", ""
    if n == "context":
        extra = f", {e['backend']} retrieval" if e.get("backend") else ""
        return f"policy mode '{e['mode']}', {e['sections']} sections{extra}", ""
    if n in ("draft", "revise"):
        text = (f"{e.get('offer_id', '-')} {e.get('discount_pct', 0)}% x {e.get('duration_months', 0)} months, "
                f"{e.get('prompt_tokens', 0)}+{e.get('completion_tokens', 0)} tokens, {e.get('latency_s', 0)}s")
        if e.get("footer_appended"):
            text += ", opt-out line appended by the system"
        if e.get("error"):
            text += f", error: {e['error']}"
        return text, ""
    if n == "verify":
        if e["passed"]:
            return "all policy rules passed", "good"
        ids = ", ".join(dict.fromkeys(rule_ids(e["violations"])))
        return f"{len(e['violations'])} violation(s): {ids}", "bad"
    if n == "finalize":
        return (f"status: {e['status']}",
                "good" if e["status"] in ("queued", "no_action") else "bad" if e["status"] in ("escalated", "error", "failed_schema") else "")
    return "", ""


def render_run(run: dict, prof: dict | None, corpus: dict) -> None:
    prop = run.get("proposal")
    left, mid, right = st.columns([1.05, 1.55, 1.25], gap="large")
    with left:
        if prof:
            ui.card("Customer", [("Contract", prof["contract"]), ("Internet", "none (phone only)" if prof["internet_service"] == "No" else prof["internet_service"]),
                                 ("Tech support", prof["tech_support"]), ("Tenure", ui.plural(prof["tenure_months"], "month")),
                                 ("Monthly charges", f"${float(prof['monthly_charges']):.2f}"),
                                 ("Marketing consent", "yes" if int(prof["marketing_opt_in"]) else "no"),
                                 ("Contacts, 90 days", str(prof["contacts_last_90d"]))])
            ui.risk_bar(float(prof["p_churn"]))
            up = json.loads(prof.get("drivers_json") or "[]")[:3]
            down = json.loads(prof.get("protective_json") or "[]")[:1]
            chips = "".join(ui.pill(f"{d['feature']} = {d['value']}", "red") for d in up) + \
                    "".join(ui.pill(f"{d['feature']} = {d['value']}", "green") for d in down)
            st.markdown(f'<div class="rc-card"><h4>Main factors behind the score</h4>{chips or "none notable"}</div>', unsafe_allow_html=True)
    with mid:
        st.markdown(ui.status_pill(run["status"]) + ui.pill(ui.plural(run.get("llm_calls", 0), "model call"), "blue") +
                    ui.pill(ui.plural(run.get("attempts", 0), "revision"), "blue"), unsafe_allow_html=True)
        if prop:
            terms = "no financial terms" if not prop["discount_pct"] and not prop["duration_months"] else \
                f"{prop['discount_pct']}% x {prop['duration_months']} months"
            ui.label("Message for the reviewer")
            ui.sms(prop["message"], f"{prop['offer_id']}  |  {terms}  |  {prop['channel']}" if prop["action"] == "OFFER" else "NO_ACTION")
            with st.expander("Why the model chose this"):
                st.markdown(prop["rationale"])
            if prop["citations"]:
                st.markdown("".join(ui.pill(c, "gray") for c in prop["citations"]), unsafe_allow_html=True)
        else:
            ui.note("The run ended before a message was drafted.")
        first = run.get("first_violations") or []
        if first:
            ui.label("What the verifier caught in the first draft")
            st.markdown("".join(ui.pill(r, "red") for r in dict.fromkeys(rule_ids(first))), unsafe_allow_html=True)
            with st.expander("Feedback sent back to the model"):
                for v in first:
                    st.markdown(f"- {v}")
        elif run["status"] not in ("skipped_gate", "error"):
            st.markdown(ui.pill("first draft passed every rule", "green"), unsafe_allow_html=True)
        if run.get("violations"):
            ui.label("Still failing after the revisions (the reviewer is warned)")
            st.markdown("".join(ui.pill(r, "red") for r in dict.fromkeys(rule_ids(run["violations"]))), unsafe_allow_html=True)
    with right:
        ui.label("Agent trace")
        for e in run.get("trace") or []:
            text, tone = trace_detail(e)
            ui.trace_step(e["node"], e["t"], text, tone)
        st.caption(f"{run.get('prompt_tokens', 0)} prompt + {run.get('completion_tokens', 0)} completion tokens, "
                   f"{run.get('llm_latency_s', 0):.1f}s of model time")
    ids = run.get("retrieved_ids") or []
    if ids:
        with st.expander(f"Policy sections the model was shown ({len(ids)})"):
            for cid in ids:
                chunk = corpus.get(cid)
                st.markdown(f"**{cid}**")
                if chunk:
                    st.markdown(f"> {chunk.text[:600]}{'...' if len(chunk.text) > 600 else ''}")


# ============================================================================ pages
def page_overview():
    m = data.metrics(str(data.path("reports")))
    reports = data.model_reports(str(data.path("reports")), "agent_eval") or data.model_reports(str(data.path("reports")), "dev_eval")
    ui.hero("Retention Copilot",
            "A calibrated churn model picks who is worth contacting. An agent drafts the message from the company's policy, "
            "a verifier checks it, and a person approves it.", eyebrow="A churn-to-action case study")
    c = st.columns(4, gap="medium")
    if m:
        ev = next((p for p in m["policies"] if p["policy"].startswith("Expected")), None)
        ui.kpi(c[0], "Churn model, ROC-AUC", f"{m['logreg']['roc_auc']:.3f}", f"held-out customers, n = {m['n_test']:,}")
        ui.kpi(c[1], "Net value per 1,000", f"${ev['net_value_per_1000']:,.0f}" if ev else "n/a", "under stated assumptions, not a forecast")
    else:
        ui.kpi(c[0], "Churn model, ROC-AUC", "n/a", "run python -m src.phase1")
        ui.kpi(c[1], "Net value per 1,000", "n/a", "")
    ui.kpi(c[2], "Policy sections", str(len(data.corpus())), "one spec writes the documents and the verifier")
    finals = [(model, cfg["configs"]["rag+verify"]) for model, cfg in reports.items() if "rag+verify" in cfg["configs"]]
    if finals:
        rates = [cfg["final_ok"]["rate"] for _, cfg in finals]
        lo, hi = min(rates), max(rates)
        value = f"{100 * hi:.0f}%" if hi - lo < 1e-9 else f"{100 * lo:.0f}\u2013{100 * hi:.0f}%"
        ui.kpi(c[3], "Drafts passing every rule", value, f"retrieval + verifier, {ui.plural(len(finals), 'model')}, n = {finals[0][1]['final_ok']['n']}")
    else:
        ui.kpi(c[3], "Drafts passing every rule", "n/a", "run python -m src.evaluate")
    st.write("")
    ui.label("How a customer moves through the system")
    steps = [("Score", "A logistic model estimates churn risk from account and billing data. Demographics are excluded."),
             ("Gate", "Expected value decides whether a contact is worth its cost. Most customers stop here."),
             ("Retrieve", "Multi-hop retrieval pulls the policy sections that apply to this customer."),
             ("Draft", "Granite writes an offer and a short SMS as schema-constrained JSON."),
             ("Verify", "A deterministic engine checks eligibility, caps, consent and wording, then sends violations back."),
             ("Approve", "A person reviews the draft. Nothing is sent automatically.")]
    cols = st.columns(6, gap="small")
    for i, (col, (t, d)) in enumerate(zip(cols, steps), 1):
        ui.step(col, i, t, d)
    st.write("")
    ui.note("Suggested route: the churn model and business case, then the agent replay, then the evaluation. "
            "The replay shows real model output, including the drafts the verifier rejected.")
    with st.expander("What this is, and what it is not"):
        st.markdown(
            "- The data is IBM's public Telco churn sample. Marketing consent and recent contacts are simulated so the compliance rules have something to test.\n"
            "- The company policy is fictional. \"Compliant\" means the draft passes my rules, not that it satisfies a regulator.\n"
            "- There is no offer versus no-offer experiment in the data, so the business-value figures rest on assumptions. The what-if tab shows how far they move.\n"
            "- The evaluation is small (48 customers per configuration), and the models ran at temperature 0 on a free GPU.")


def calibration_frame(df: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
    d = df.copy()
    d["bin"] = pd.qcut(d["p_churn"], bins, duplicates="drop")
    return d.groupby("bin", observed=True).agg(predicted=("p_churn", "mean"), observed=("churn_actual", "mean"),
                                               customers=("p_churn", "size")).reset_index(drop=True)


def page_model():
    m = data.metrics(str(data.path("reports")))
    ui.heading("Churn model and business case", "Who is at risk, who is worth contacting, and how much that answer depends on my assumptions.")
    if not m:
        ui.note("No model report found. Run python -m src.phase1 first.")
        return
    try:
        df = data.test_frame(str(data.path("db")))
    except Exception:
        df = None
    tab1, tab2, tab3 = st.tabs(["Model", "Business case", "What-if economics"])
    with tab1:
        rows = []
        for key, label in (("logreg", "Logistic regression (shipped)"), ("hist_gb", "Gradient boosting (benchmark)")):
            r = m[key]
            rows.append({"Model": label, "CV ROC-AUC": r["cv_roc_auc"], "Test ROC-AUC": r["roc_auc"], "PR-AUC": r["pr_auc"],
                         "Brier": r["brier"], "ECE": r["ece"], "Top-decile lift": r["top_decile_lift"]})
        st.dataframe(pd.DataFrame(rows).round(3), hide_index=True, width="stretch")
        d = m["delta_auc_gb_minus_lr"]
        ui.note(f"Paired bootstrap, gradient boosting minus logistic: ROC-AUC {d['mean']:+.4f} (95% CI {d['lo']:+.4f} to {d['hi']:+.4f}). "
                "The interval includes zero, so I shipped the model I can explain customer by customer.")
        a, b = st.columns(2, gap="large")
        with a:
            ui.label("Calibration on held-out customers")
            if df is not None:
                cal = calibration_frame(df)
                diag = alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]})).mark_line(color="#BFB3A3", strokeDash=[5, 4]).encode(x="x:Q", y="y:Q")
                line = alt.Chart(cal).mark_line(color=BRONZE, point=alt.OverlayMarkDef(color=BRONZE, size=60)).encode(
                    x=alt.X("predicted:Q", title="Predicted churn probability", scale=alt.Scale(domain=[0, 1])),
                    y=alt.Y("observed:Q", title="Observed churn rate", scale=alt.Scale(domain=[0, 1])),
                    tooltip=[alt.Tooltip("predicted:Q", format=".2f"), alt.Tooltip("observed:Q", format=".2f"), "customers"])
                st.altair_chart(styled(diag + line, 300), width="stretch")
        with b:
            ui.label("Churn falls steeply with tenure, so the model uses tenure bands")
            try:
                t = data.tenure_table(str(data.path("db")))
                chart = alt.Chart(t).mark_bar(color=BRONZE, size=34).encode(
                    x=alt.X("band:N", sort=list(t["band"]), title="Tenure (months)", axis=alt.Axis(labelAngle=0)),
                    y=alt.Y("churn_rate:Q", title="Churn rate", axis=alt.Axis(format="%")),
                    tooltip=["band", "customers", alt.Tooltip("churn_rate:Q", format=".1%")])
                st.altair_chart(styled(chart, 300), width="stretch")
            except Exception:
                pass
    with tab2:
        pol = pd.DataFrame(m["policies"])
        pol = pd.DataFrame({"Policy": pol["policy"], "Contacted": pol["contacted_pct"].map("{:.0f}%".format),
                            "Churners reached": pol["churners_reached_pct"].map("{:.0f}%".format),
                            "Precision": pol["precision_pct"].map("{:.0f}%".format),
                            "Net value per 1,000": pol["net_value_per_1000"].map("${:,.0f}".format)})
        st.dataframe(pol, hide_index=True, width="stretch")
        ui.note("Contacting a customer only when expected value is positive beats both contacting everyone and a fixed top-20% list. "
                "How much it wins by depends on the assumptions, which is what the grid below shows.")
        if df is not None:
            base = Economics(**{k: m["economics"][k] for k in Economics.__dataclass_fields__})
            grid = sensitivity_grid(df["p_churn"].to_numpy(), df["churn_actual"].to_numpy(), df["monthly_charges"].to_numpy(), base,
                                    [0.10, 0.15, 0.20, 0.25, 0.30, 0.40], [0.05, 0.10, 0.15, 0.20, 0.25])
            grid["save"] = (grid["save_rate"] * 100).round().astype(int).astype(str) + "%"
            grid["disc"] = (grid["discount_pct"] * 100).round().astype(int).astype(str) + "%"
            grid["label"] = grid["net_value_per_1000"].map("${:,.0f}".format)
            ui.label("Net value per 1,000 customers as the assumptions change")
            heat = alt.Chart(grid).mark_rect(stroke="#FAF7F2", strokeWidth=2).encode(
                x=alt.X("disc:N", sort=[f"{d}%" for d in (5, 10, 15, 20, 25)], title="Discount depth", axis=alt.Axis(labelAngle=0)),
                y=alt.Y("save:N", sort=[f"{s}%" for s in (10, 15, 20, 25, 30, 40)], title="Assumed save rate"),
                color=alt.Color("net_value_per_1000:Q", scale=alt.Scale(range=["#F2ECE3", "#D9C09A", "#8C6A3F"]), legend=None),
                tooltip=["save", "disc", alt.Tooltip("net_value_per_1000:Q", format=",.0f"), alt.Tooltip("contacted_pct:Q", format=".0f")])
            text = alt.Chart(grid).mark_text(fontSize=13, color="#2A2420").encode(x="disc:N", y="save:N", text="label:N")
            st.altair_chart(styled(heat + text, 330), width="stretch")
        econ = m["economics"]
        with st.expander("Assumptions behind these numbers"):
            st.dataframe(pd.DataFrame({"Assumption": ["Save rate", "Discount taken by customers who would have stayed", "Offer", "Margin",
                                                       "Extra lifetime months if saved", "Cost per contact"],
                                       "Value": [f"{econ['save_rate']:.0%}", f"{econ['takeup_non_churners']:.0%}",
                                                 f"{econ['discount_pct']:.0%} off for {econ['discount_months']} months", f"{econ['margin']:.0%}",
                                                 str(econ["saved_lifetime_months"]), f"${econ['contact_cost']:.2f}"]}),
                         hide_index=True, width="stretch")
    with tab3:
        if df is None:
            ui.note("The analytics database is missing. Run python -m src.phase1.")
            return
        base = Economics(**{k: m["economics"][k] for k in Economics.__dataclass_fields__})
        st.markdown("Move a slider and the contact policy is re-evaluated on the held-out customers. "
                    "Decisions use predicted risk only; outcomes use the real labels plus your assumed treatment effect.")
        s1, s2, s3 = st.columns(3, gap="large")
        save = s1.slider("Save rate", 0.05, 0.60, float(base.save_rate), 0.01, help="Share of would-be churners kept by an offer")
        take = s2.slider("Discount taken by customers who would have stayed", 0.0, 0.8, float(base.takeup_non_churners), 0.01)
        disc = s3.slider("Discount depth", 0.05, 0.30, float(base.discount_pct), 0.01)
        s4, s5, s6 = st.columns(3, gap="large")
        months = s4.slider("Discount months", 1, 12, int(base.discount_months))
        margin = s5.slider("Margin on monthly charges", 0.15, 0.60, float(base.margin), 0.01)
        cost = s6.slider("Cost per contact ($)", 0.0, 15.0, float(base.contact_cost), 0.5)
        econ = replace(base, save_rate=save, takeup_non_churners=take, discount_pct=disc, discount_months=months, margin=margin, contact_cost=cost)
        p, y, mrr = df["p_churn"].to_numpy(), df["churn_actual"].to_numpy(), df["monthly_charges"].to_numpy()
        pol = policy_comparison(p, y, mrr, econ)
        prop = pol[pol["policy"].str.startswith("Expected")].iloc[0]
        st.write("")
        k1, k2, k3 = st.columns(3, gap="medium")
        ui.kpi(k1, "Expected-value policy", f"${prop['net_value_per_1000']:,.0f}", f"per 1,000 customers, {prop['contacted_pct']:.0f}% contacted")
        top = pol[pol["policy"].str.startswith("Top")].iloc[0]
        ui.kpi(k2, "Top 20% by risk", f"${top['net_value_per_1000']:,.0f}", "fixed-size list")
        allp = pol[pol["policy"] == "Contact everyone"].iloc[0]
        ui.kpi(k3, "Contact everyone", f"${allp['net_value_per_1000']:,.0f}", "no targeting")
        curve = profit_curve(p, y, mrr, econ)
        pts = pd.DataFrame({"pct_contacted": [prop["contacted_pct"]], "net_value_per_1000": [prop["net_value_per_1000"]]})
        line = alt.Chart(curve).mark_line(color=BRONZE, strokeWidth=2.5).encode(
            x=alt.X("pct_contacted:Q", title="% of customers contacted, best expected value first"),
            y=alt.Y("net_value_per_1000:Q", title="Net value per 1,000 customers ($)"))
        dot = alt.Chart(pts).mark_point(color=TERRACOTTA, size=150, filled=True).encode(x="pct_contacted:Q", y="net_value_per_1000:Q")
        zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="#BFB3A3", strokeDash=[4, 4]).encode(y="y:Q")
        st.altair_chart(styled(line + dot + zero, 330), width="stretch")


def ollama_models(url: str = "http://127.0.0.1:11434") -> list[str]:
    try:
        with urllib.request.urlopen(f"{url}/api/tags", timeout=0.6) as r:
            return [m["name"] for m in json.load(r).get("models", [])]
    except Exception:
        return []


def page_agent():
    ui.heading("Agent replay", "Real model output from the evaluation runs. Pick a customer and read the whole decision.")
    sets = data.run_sets()
    corpus = data.corpus()
    tab1, tab2, tab3 = st.tabs(["Decision", "Same customer, every configuration", "Run live"])
    if not sets:
        with tab1:
            ui.note("No agent runs found yet. Run python -m src.evaluate (see the README) and reload.")
    else:
        c1, c2, c3 = st.columns(3)
        rs = c1.selectbox("Run set", sets, format_func=lambda s: s["label"])
        models = rs["models"]
        model = c2.selectbox("Model", models, index=models.index("granite4:small-h") if "granite4:small-h" in models else len(models) - 1,
                             format_func=model_name)
        all_runs = data.runs(rs["path"], model)
        cfgs = [c for c in CONFIG_ORDER if c in {r["config"] for r in all_runs}]
        cfg = c3.selectbox("Configuration", cfgs, index=cfgs.index("rag+verify") if "rag+verify" in cfgs else 0,
                           format_func=lambda c: f"{c}  ({CONFIG_HELP[c]})")
        pick = [r for r in all_runs if r["config"] == cfg]

        def rank(r):                       # repaired drafts first: they show the verifier doing its job
            if r["status"] == "queued" and r["attempts"] > 0:
                return 0
            return {"escalated": 1, "queued": 2, "queued_unverified": 3, "no_action": 4}.get(r["status"], 9)
        pick.sort(key=lambda r: (rank(r), r["customer_id"]))
        labels = {r["customer_id"]: f"{r['customer_id']}   ·   {r['status']}, {ui.plural(r['attempts'], 'revision')}" for r in pick}
        with tab1:
            cid = st.selectbox("Customer", list(labels), format_func=labels.get)
            run = next(r for r in pick if r["customer_id"] == cid)
            render_run(run, data.profile(str(data.path("db")), cid), corpus)
        with tab2:
            st.markdown(f"Customer **{cid}**, {model_name(model)}: the same customer under every configuration that was run.")
            rows = []
            for r in sorted([r for r in all_runs if r["customer_id"] == cid], key=lambda r: CONFIG_ORDER.index(r["config"])):
                p = r["proposal"] or {}
                rows.append({"Configuration": r["config"], "Status": r["status"], "Offer": p.get("offer_id", "-"),
                             "Terms": f"{p.get('discount_pct', 0)}% x {p.get('duration_months', 0)} mo",
                             "Rules broken (first draft)": ", ".join(dict.fromkeys(rule_ids(r["first_violations"]))) or "none",
                             "Rules broken (final)": ", ".join(dict.fromkeys(rule_ids(r["violations"]))) or "none",
                             "Message": p.get("message", "")})
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    with tab3:
        up = ollama_models()
        if not up:
            ui.note("No local Ollama server detected on port 11434. Live runs need one (ollama serve, then ollama pull granite4:micro). "
                    "The replays above do not.")
        else:
            st.success(f"Ollama detected with {len(up)} model(s).")
            l1, l2, l3 = st.columns(3)
            model_l = l1.selectbox("Model", up, key="live_model")
            cfg_l = l2.selectbox("Configuration", CONFIG_ORDER, index=CONFIG_ORDER.index("rag+verify"), key="live_cfg")
            cid_l = l3.text_input("Customer id", value="7590-VHVEG", key="live_cid")
            if st.button("Run the agent now"):
                from src.agent import CONFIGS, RetentionAgent           # imported lazily: needs langgraph
                from src.sql_tool import SqlTool
                with st.spinner("Running..."):
                    try:
                        agent = RetentionAgent(replace(CONFIGS[cfg_l], model=model_l, enqueue=False, retrieval_backend="tfidf"),
                                               sql=SqlTool(data.path("db")), persist=False)
                        out = agent.run(cid_l.strip())
                        out["model"] = model_l
                        render_run(out, data.profile(str(data.path("db")), cid_l.strip()), corpus)
                    except Exception as exc:                            # show the problem instead of a stack trace
                        st.error(f"{type(exc).__name__}: {exc}")


def page_queue():
    ui.heading("Review queue", "The last step is a person. Approve or reject each draft; the decision is stored with a note.")
    store = RunStore(data.path("runs"))
    sets = data.run_sets()
    with st.expander("Load sample drafts from the evaluation runs", expanded=not store.queue(None)):
        n = st.slider("How many drafts", 3, 20, 8)
        if st.button("Load into the queue", disabled=not sets):
            rs = sets[0]
            models = rs["models"]
            model = "granite4:small-h" if "granite4:small-h" in models else models[-1]
            existing = {q["customer_id"] for q in store.queue(None)}
            added = 0
            for r in data.runs(rs["path"], model):
                if r["config"] != "rag+verify" or r["customer_id"] in existing or not r["proposal"]:
                    continue
                if r["proposal"]["action"] != "OFFER" or r["status"] not in ("queued", "escalated"):
                    continue
                prof = data.profile(str(data.path("db")), r["customer_id"])
                flags = review_flags(prof, Proposal(**r["proposal"])) + (["verifier_failed"] if r["status"] == "escalated" else [])
                store.enqueue(r["run_id"], r["customer_id"], r["proposal"], flags, r["violations"] or [])
                added += 1
                if added >= n:
                    break
            st.success(f"Added {added} draft(s) from {model_name(model)}.")
            st.rerun()
        if not sets:
            st.caption("Evaluation runs are needed first.")
    pending, decided = store.queue("pending"), store.queue("approved") + store.queue("rejected")
    t1, t2 = st.tabs([f"Pending ({len(pending)})", f"Decided ({len(decided)})"])
    tones = {"high_value_account": "amber", "new_customer": "blue", "verifier_failed": "red"}
    with t1:
        if not pending:
            ui.note("Nothing waiting. Load some drafts above.")
        for item in pending:
            prop, flags = json.loads(item["proposal_json"]), json.loads(item["flags_json"])
            with st.container(border=True):
                a, b, c = st.columns([1.0, 1.7, 1.1], gap="medium")
                with a:
                    st.markdown(f"**{item['customer_id']}**")
                    st.markdown(f"{prop['offer_id']}  \n{prop['discount_pct']}% x {prop['duration_months']} months")
                    st.markdown("".join(ui.pill(f.replace("_", " "), tones.get(f, "gray")) for f in flags) or ui.pill("no flags", "green"),
                                unsafe_allow_html=True)
                with b:
                    ui.sms(prop["message"], f"{prop['channel']}")
                    if json.loads(item["violations_json"]):
                        st.warning("Verifier still reports: " + ", ".join(dict.fromkeys(rule_ids(json.loads(item["violations_json"])))))
                with c:
                    note = st.text_input("Reviewer note", key=f"note_{item['queue_id']}")
                    ok, no = st.columns(2)
                    if ok.button("Approve", key=f"ok_{item['queue_id']}", type="primary", width="stretch"):
                        store.decide(item["queue_id"], "approved", note)
                        st.rerun()
                    if no.button("Reject", key=f"no_{item['queue_id']}", width="stretch"):
                        store.decide(item["queue_id"], "rejected", note)
                        st.rerun()
    with t2:
        if decided:
            rows = [{"Customer": d["customer_id"], "Decision": d["status"], "Note": d["reviewer_note"], "When (UTC)": d["decided_at"],
                     "Message": json.loads(d["proposal_json"])["message"]} for d in decided]
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        else:
            ui.note("No decisions yet.")


def bars(rep: dict, key: str, stages: list[tuple[str, str]], colours: list[str]):
    rows = []
    for cfg in CONFIG_ORDER:
        c = rep["configs"].get(cfg)
        if c:
            for stage, k in stages:
                rows.append({"config": cfg, "stage": stage, "rate": c[k]["rate"], "lo": c[k]["ci"][0], "hi": c[k]["ci"][1]})
    df = pd.DataFrame(rows)
    order = [c for c in CONFIG_ORDER if c in df["config"].unique()]
    names = [s for s, _ in stages]
    base = alt.Chart(df).encode(x=alt.X("config:N", sort=order, title=None, axis=alt.Axis(labelAngle=0)))
    offset = alt.XOffset("stage:N", sort=names)
    bar = base.mark_bar(size=24 if len(names) > 1 else 40).encode(
        y=alt.Y("rate:Q", title=None, axis=alt.Axis(format="%", tickCount=5), scale=alt.Scale(domain=[0, 1.05])), xOffset=offset,
        color=alt.Color("stage:N", scale=alt.Scale(domain=names, range=colours), legend=alt.Legend(title=None, orient="top") if len(names) > 1 else None),
        tooltip=["config", "stage", alt.Tooltip("rate:Q", format=".0%")])
    err = base.mark_errorbar(color=ESPRESSO, thickness=1.2).encode(y=alt.Y("lo:Q", title=None), y2="hi:Q", xOffset=offset)
    return styled(bar + err, 300)


def page_eval():
    rep_dir = str(data.path("reports"))
    final, dev = data.model_reports(rep_dir, "agent_eval"), data.model_reports(rep_dir, "dev_eval")
    reports, kind = (final, "Final") if final else (dev, "Development")
    ui.heading("Evaluation", "The same held-out customers, with the policy given to the model in different ways.")
    if not reports:
        ui.note("No evaluation report found. Run python -m src.evaluate (see the README).")
        return
    if kind == "Development":
        ui.note("These are development runs used to tune prompts. The final held-out evaluation has not been added yet.")
    names = list(reports)
    model = st.radio("Model", names, index=names.index("granite4-small-h") if "granite4-small-h" in names else 0,
                     format_func=model_name, horizontal=True)
    m = reports[model]
    a, b = st.columns(2, gap="large")
    with a:
        ui.label("Drafts passing every policy rule")
        st.altair_chart(bars(m, "first_ok", [("first draft", "first_ok"), ("final", "final_ok")], [SANDBAR, BRONZE]), width="stretch")
    with b:
        ui.label("Offer matches the segment playbook")
        st.altair_chart(bars(m, "matches_preferred", [("final offer", "matches_preferred")], [BRONZE]), width="stretch")
    st.caption(f"{kind} evaluation: {m['n']} customers per configuration, identical for every bar. Error bars are 95% Wilson intervals. "
               "A draft that went through the verifier loop is compliant almost by construction, so the first-draft bars show the effect of the "
               "policy text and the model; the playbook bars show whether the decision was right.")
    rows = []
    for cfg in CONFIG_ORDER:
        c = m["configs"].get(cfg)
        if c:
            rows.append({"Configuration": cfg, "First draft compliant": rate_text(c["first_ok"]), "Final compliant": rate_text(c["final_ok"]),
                         "Offer matches playbook": rate_text(c["matches_preferred"]), "Prompt tokens": round(c["mean_prompt_tokens"]),
                         "Model time / run (s)": round(c["mean_llm_latency_s"], 1)})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    with st.expander("Paired comparisons on the same customers"):
        con = pd.DataFrame(m["contrasts"]).rename(columns={"contrast": "Paired comparison", "only_first": "Only first succeeds",
                                                          "only_second": "Only second succeeds", "p": "Exact McNemar p"})
        con["Exact McNemar p"] = con["Exact McNemar p"].map(lambda v: "< 0.0001" if v < 0.0001 else f"{v:.4f}")
        st.dataframe(con, hide_index=True, width="stretch")
    with st.expander("Which rules the first drafts break (share of customers)"):
        rules = pd.DataFrame(m["rule_rates"]).fillna(0.0)
        if not rules.empty:
            st.dataframe(rules.style.format("{:.0%}").background_gradient(cmap="YlOrBr", vmin=0, vmax=1), width="stretch")
    with st.expander("Both models side by side"):
        both = []
        for name, rep in reports.items():
            for cfg in CONFIG_ORDER:
                c = rep["configs"].get(cfg)
                if c:
                    both.append({"Model": model_name(name), "Configuration": cfg, "First draft compliant": rate_text(c["first_ok"]),
                                 "Final compliant": rate_text(c["final_ok"]), "Offer matches playbook": rate_text(c["matches_preferred"]),
                                 "Model calls": round(c["mean_llm_calls"], 2)})
        st.dataframe(pd.DataFrame(both), hide_index=True, width="stretch")


def page_policy():
    ui.heading("Policy and retrieval", "The fictional company policy the agent must follow, and how well it can be found.")
    docs = sorted((ROOT / "policies").glob("*.md"))
    t1, t2, t3 = st.tabs(["Policy documents", "Try retrieval", "Retrieval benchmark"])
    with t1:
        ui.note("These files are generated from one spec (src/policy_spec.py). The verifier reads the same constants, and a test fails if the text and the rules drift apart.")
        for d in docs:
            with st.expander(d.stem.replace("_", " ").title()):
                st.markdown(d.read_text(encoding="utf-8"))
    with t2:
        q = st.text_input("Ask the policy a question", "Can a customer with no marketing consent get a discount?")
        k = st.slider("Sections to show", 1, 8, 4)
        if q.strip():
            for chunk, score in data.retriever("tfidf").search(q, k=k):
                st.markdown(f"**{chunk.id}**  " + ui.pill(f"score {score:.2f}", "blue"), unsafe_allow_html=True)
                st.markdown(f"> {chunk.text[:420]}{'...' if len(chunk.text) > 420 else ''}")
        st.caption("This box uses TF-IDF. The agent itself uses hybrid retrieval (TF-IDF + dense embeddings) when the model is available.")
    with t3:
        rep = data.retrieval_report(str(data.path("reports")))
        if not rep:
            ui.note("Run python -m src.retrieval_eval to compare TF-IDF, dense and hybrid retrieval on 28 paraphrased questions.")
        else:
            df = pd.DataFrame([{k: v for k, v in r.items() if k != "misses_at_3"} for r in rep]).rename(columns={"backend": "Backend", "queries": "Queries"})
            st.dataframe(df.round(2), hide_index=True, width="stretch")
            st.caption("hit@k: the right section is among the top k results. MRR: mean reciprocal rank of the first right section.")


PAGES = {"Overview": page_overview, "Churn model and business case": page_model, "Agent replay": page_agent,
         "Review queue": page_queue, "Evaluation": page_eval, "Policy and retrieval": page_policy}

with st.sidebar:
    ui.wordmark()
    page = st.radio("Navigate", list(PAGES), label_visibility="collapsed")
    st.markdown("---")
    st.markdown(f"[Source code on GitHub]({REPO_URL})")
    st.caption("Public IBM Telco churn sample. Company policy and two customer fields are simulated.")

PAGES[page]()
ui.footer("Portfolio project. IBM and Granite are trademarks of their owners; this project is not affiliated with or endorsed by IBM.")
