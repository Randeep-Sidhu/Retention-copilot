"""Streamlit app: every page renders, the review queue works, the what-if page recomputes (headless, offline)."""
import json
from dataclasses import replace
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from src.agent import CONFIGS, RetentionAgent
from src.evaluate import report
from src.llm import ScriptedLLM
from src.retriever import Retriever, load_corpus
from src.sql_tool import SqlTool
from src.store import RunStore

APP = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")
IDS = ["A-FIBER", "B-ONEYR", "C-OPTOUT", "D-LIMIT", "F-2YR"]
PAGES = ["Overview", "Churn model and business case", "Agent replay", "Review queue", "Evaluation", "Policy and retrieval"]

METRICS = {
    "n_customers": 7043, "churn_rate": 0.265, "n_train": 5282, "n_test": 1761, "selected_model": "logreg",
    "logreg": {"roc_auc": 0.841, "pr_auc": 0.645, "brier": 0.137, "ece": 0.018, "top_decile_lift": 2.79, "cv_roc_auc": 0.85},
    "hist_gb": {"roc_auc": 0.843, "pr_auc": 0.658, "brier": 0.136, "ece": 0.012, "top_decile_lift": 2.85, "cv_roc_auc": 0.85},
    "delta_auc_gb_minus_lr": {"mean": 0.0024, "lo": -0.0021, "hi": 0.0068},
    "economics": {"save_rate": 0.25, "takeup_non_churners": 0.25, "discount_pct": 0.15, "discount_months": 6, "margin": 0.35,
                  "saved_lifetime_months": 12, "contact_cost": 3.0},
    "policies": [{"policy": "Contact nobody", "contacted_pct": 0, "churners_reached_pct": 0, "precision_pct": 0, "net_value_per_1000": 0},
                 {"policy": "Contact everyone", "contacted_pct": 100, "churners_reached_pct": 100, "precision_pct": 26.5, "net_value_per_1000": 3406},
                 {"policy": "Top 20% by churn risk", "contacted_pct": 20, "churners_reached_pct": 50, "precision_pct": 66, "net_value_per_1000": 6805},
                 {"policy": "Expected value > 0 (proposed)", "contacted_pct": 40, "churners_reached_pct": 77, "precision_pct": 50, "net_value_per_1000": 8990}],
}


class NoisyLLM(ScriptedLLM):
    """First drafts carry an extra number; revisions are clean (so the verifier loop has something to repair)."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def draft(self, messages, profile=None, attempt=0):
        self.always_fault = (lambda p: p.model_copy(update={"message": p.message + " Ends in 5 days."})) if len(messages) == 2 else None
        return super().draft(messages, profile, attempt)


@pytest.fixture()
def demo(test_db, tmp_path, monkeypatch):
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "metrics.json").write_text(json.dumps(METRICS))
    runs_db = tmp_path / "eval_runs.db"
    for model in ("granite4:micro", "granite4:small-h"):
        for name in ("none", "rag", "rag+verify"):
            agent = RetentionAgent(replace(CONFIGS[name], enqueue=False), llm=NoisyLLM(model), sql=SqlTool(test_db),
                                   store=RunStore(runs_db), retriever=Retriever(load_corpus(), "tfidf"))
            for cid in IDS:
                agent.run(cid)
        report(RunStore(runs_db), model, db_path=test_db, out_dir=reports)
    for key, value in {"RC_DB": test_db, "RC_EVAL_DB": runs_db, "RC_DEV_DB": tmp_path / "none_dev.db",
                       "RC_RUNS_DB": tmp_path / "queue.db", "RC_REPORTS": reports}.items():
        monkeypatch.setenv(key, str(value))
    return tmp_path


def open_page(name):
    at = AppTest.from_file(APP, default_timeout=90)
    at.run()
    assert not at.exception
    at.sidebar.radio[0].set_value(name).run()
    return at


def html_of(at) -> str:
    return " ".join(m.value for m in at.markdown)


@pytest.mark.parametrize("page", PAGES)
def test_every_page_renders_with_demo_data(demo, page):
    at = open_page(page)
    assert not at.exception, [e.value for e in at.exception]


@pytest.mark.parametrize("page", PAGES)
def test_every_page_renders_with_no_data_at_all(tmp_path, monkeypatch, page):
    for key in ("RC_DB", "RC_EVAL_DB", "RC_DEV_DB", "RC_RUNS_DB", "RC_REPORTS"):
        monkeypatch.setenv(key, str(tmp_path / f"missing_{key}"))
    at = open_page(page)
    assert not at.exception, [e.value for e in at.exception]


def test_agent_replay_shows_the_message_the_verifier_feedback_and_the_trace(demo):
    at = open_page("Agent replay")
    text = html_of(at)
    assert "Message for the reviewer" in text and "rc-sms" in text and "What the verifier caught" in text
    assert "rc-trace" in text and "Agent trace" in text


def test_overview_reports_the_final_pass_rate_from_the_evaluation(demo):
    text = html_of(open_page("Overview"))
    assert "0.841" in text and "8,990" in text and "Drafts passing every rule" in text and "100%" in text


def test_review_queue_load_then_approve_is_recorded(demo):
    at = open_page("Review queue")
    next(b for b in at.button if b.label == "Load into the queue").click().run()
    store = RunStore(demo / "queue.db")
    pending = store.queue("pending")
    assert pending and all(json.loads(p["flags_json"]) is not None for p in pending)
    qid = pending[0]["queue_id"]
    at = open_page("Review queue")
    at.text_input(key=f"note_{qid}").set_value("fine").run()
    at.button(key=f"ok_{qid}").click().run()
    assert store.queue("approved")[0]["reviewer_note"] == "fine"
    assert len(store.queue("pending")) == len(pending) - 1


def test_what_if_page_recomputes_when_an_assumption_changes(demo):
    at = open_page("Churn model and business case")
    before = html_of(at)
    at.slider[0].set_value(0.55).run()          # save rate
    assert not at.exception
    assert html_of(at) != before


def test_evaluation_page_has_a_model_switch_charts_and_the_tables(demo):
    at = open_page("Evaluation")
    assert not at.exception, [e.value for e in at.exception]
    radio = next(r for r in at.radio if r.label == "Model")
    assert set(radio.options) == {"Granite 4 Micro (3B)", "Granite 4 H-Small (32B MoE)"}
    radio.set_value("Granite 4 Micro (3B)").run()
    assert not at.exception and len(at.dataframe) >= 1


def test_overview_has_no_technology_chip_row(demo):
    text = html_of(open_page("Overview"))
    assert "rc-tag" not in text and "scikit-learn" not in text
