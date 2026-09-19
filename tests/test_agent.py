"""LangGraph agent: routing, repair loop, escalation, monitoring mode, queue and run log (scripted LLM, offline)."""
import warnings

import pytest

from src.agent import CONFIGS, RetentionAgent, conditional_queries, retrieve_context
from src.llm import ScriptedLLM
from src.policy_engine import required_sections
from src.prompts import build_messages
from src.retriever import Retriever, load_corpus
from src.sql_tool import SqlTool
from src.store import RunStore

warnings.filterwarnings("ignore")
NODES_HAPPY = ["load_customer", "gate", "context", "draft", "verify", "finalize"]


def make_agent(test_db, tmp_path, cfg="rag+verify", llm=None, **overrides):
    from dataclasses import replace
    config = replace(CONFIGS[cfg], **overrides) if overrides else CONFIGS[cfg]
    return RetentionAgent(config, llm=llm or ScriptedLLM(), sql=SqlTool(test_db), store=RunStore(tmp_path / "runs.db"),
                          retriever=Retriever(load_corpus(), "tfidf"))


def drop_opt_out(p):
    return p.model_copy(update={"message": p.message.replace(" Reply STOP to opt out.", "")})


def too_big(p):
    return p.model_copy(update={"discount_pct": 25, "offer_id": "LOYALTY_DISCOUNT", "duration_months": 6,
                                "message": "Hello, enjoy 25% off for 6 months. Reply STOP to opt out.",
                                "citations": ["offer_catalog#loyalty-discount"]})


def test_happy_path_routes_through_every_node_and_queues_for_review(test_db, tmp_path):
    agent = make_agent(test_db, tmp_path)
    out = agent.run("A-FIBER")
    assert [e["node"] for e in out["trace"]] == NODES_HAPPY
    assert out["status"] == "queued" and out["proposal"]["offer_id"] == "TECH_SUPPORT_TRIAL"
    assert out["llm_calls"] == 1 and out["attempts"] == 0 and out["violations"] == []
    q = agent.store.queue()
    assert len(q) == 1 and q[0]["customer_id"] == "A-FIBER" and q[0]["status"] == "pending"
    assert agent.store.runs()[0]["status"] == "queued"


def test_low_risk_customer_is_skipped_by_the_expected_value_gate(test_db, tmp_path):
    agent = make_agent(test_db, tmp_path)
    out = agent.run("E-LOWRISK")
    assert out["status"] == "skipped_gate" and out["llm_calls"] == 0
    assert [e["node"] for e in out["trace"]] == ["load_customer", "gate", "finalize"]
    assert agent.store.queue() == []


def test_unknown_customer_is_an_error_not_a_crash(test_db, tmp_path):
    out = make_agent(test_db, tmp_path).run("NOPE")
    assert out["status"] == "error" and out["llm_calls"] == 0


def test_verifier_feedback_repairs_a_bad_first_draft(test_db, tmp_path):
    llm = ScriptedLLM(faults=[drop_opt_out])
    out = make_agent(test_db, tmp_path, llm=llm).run("A-FIBER")
    assert any("opt_out_line" in v for v in out["first_violations"])
    assert out["violations"] == [] and out["attempts"] == 1 and out["llm_calls"] == 2 and out["status"] == "queued"
    assert [e["node"] for e in out["trace"]].count("revise") == 1


def test_escalates_after_max_revisions_with_a_flag_for_the_reviewer(test_db, tmp_path):
    agent = make_agent(test_db, tmp_path, llm=ScriptedLLM(always_fault=drop_opt_out))
    out = agent.run("A-FIBER")
    assert out["status"] == "escalated" and out["attempts"] == 2 and out["llm_calls"] == 3
    assert "verifier_failed" in agent.store.queue()[0]["flags_json"]


def test_verify_off_only_monitors_and_never_revises(test_db, tmp_path):
    out = make_agent(test_db, tmp_path, cfg="rag", llm=ScriptedLLM(faults=[too_big])).run("B-ONEYR")
    assert out["llm_calls"] == 1 and out["attempts"] == 0
    assert out["status"] == "queued_unverified"
    assert any("discount_cap" in v for v in out["violations"]) and out["first_violations"] == out["violations"]


def test_contact_limit_yields_no_action_and_nothing_is_queued(test_db, tmp_path):
    agent = make_agent(test_db, tmp_path)
    out = agent.run("D-LIMIT")
    assert out["proposal"]["action"] == "NO_ACTION" and out["status"] == "no_action" and out["violations"] == []
    assert agent.store.queue() == []


@pytest.mark.parametrize("cfg", list(CONFIGS))
def test_every_config_runs_end_to_end(test_db, tmp_path, cfg):
    agent = make_agent(test_db, tmp_path, cfg=cfg)
    for cid in ("A-FIBER", "B-ONEYR", "C-OPTOUT", "D-LIMIT", "F-2YR"):
        out = agent.run(cid)
        assert out["status"] in ("queued", "queued_unverified", "no_action"), (cfg, cid, out["violations"])


def test_human_review_decision_is_recorded(test_db, tmp_path):
    agent = make_agent(test_db, tmp_path)
    agent.run("A-FIBER")
    qid = agent.store.queue()[0]["queue_id"]
    agent.store.decide(qid, "approved", "looks good")
    assert agent.store.queue("pending") == [] and agent.store.queue("approved")[0]["reviewer_note"] == "looks good"
    with pytest.raises(ValueError):
        agent.store.decide(qid, "maybe")


# ------------------------------------------------------------------ prompts and retrieval context
def test_prompts_differ_by_policy_mode_and_revision_carries_the_violations(test_db):
    prof = SqlTool(test_db).profile("A-FIBER")
    corpus = load_corpus()
    from src.prompts import format_chunks
    none = build_messages("none", "", prof)
    full = build_messages("full", format_chunks(corpus), prof)
    assert "offer_catalog#loyalty-discount" not in none[0][1] and "offer_catalog#loyalty-discount" in full[0][1]
    assert "churn_risk_score: 0.85" in none[1][1] and "tenure_band" in none[1][1]
    rev = build_messages("rag", "", prof, previous_json="{}", violations=["[numbers] fix it"])
    assert [r for r, _ in rev] == ["system", "human", "ai", "human"] and "[numbers] fix it" in rev[-1][1]


def test_conditional_queries_adapt_to_the_customer(test_db):
    sql = SqlTool(test_db)
    assert any("contact frequency" in q for q in conditional_queries(sql.profile("D-LIMIT")))
    assert any("marketing consent" in q for q in conditional_queries(sql.profile("C-OPTOUT")))
    assert not any("contact frequency" in q for q in conditional_queries(sql.profile("A-FIBER")))


def test_multi_hop_retrieval_fetches_the_catalog_sections_named_by_the_playbook(test_db):
    ret = Retriever(load_corpus(), "tfidf")
    ids = {c.id for c in retrieve_context(ret, SqlTool(test_db).profile("A-FIBER"))}
    assert "segment_playbooks#month-to-month-fiber-optic" in ids
    assert {"offer_catalog#tech-support-trial", "offer_catalog#term-upgrade", "offer_catalog#loyalty-discount",
            "offer_catalog#service-check-in", "discount_caps#risk-tiers"} <= ids
    assert len(ids) < len(load_corpus())            # still a strict subset of the corpus


def test_rag_context_recalls_most_required_sections(test_db, tmp_path):
    agent = make_agent(test_db, tmp_path)
    sql, recalls = SqlTool(test_db), []
    for cid in ("A-FIBER", "B-ONEYR", "C-OPTOUT", "D-LIMIT", "F-2YR"):
        prof = sql.profile(cid)
        got = set(agent.run(cid)["retrieved_ids"])
        need = required_sections(prof)
        recalls.append(len(need & got) / len(need))
    assert sum(recalls) / len(recalls) >= 0.9, recalls


def test_policy_modes_get_the_decision_procedure_and_baseline_does_not(test_db):
    prof = SqlTool(test_db).profile("A-FIBER")
    for mode, has in (("none", False), ("full", True), ("rag", True)):
        system = build_messages(mode, "[x#y]\ntext" if mode != "none" else "", prof)[0][1]
        assert ("FIRST offer in its preferred order" in system) is has, mode
        assert "short SMS" in system                                    # format guidance is shared by every config


def test_revisions_are_requested_with_increasing_attempt_numbers(test_db, tmp_path):
    seen = []

    class Spy(ScriptedLLM):
        def draft(self, messages, profile=None, attempt=0):
            seen.append(attempt)
            return super().draft(messages, profile, attempt)

    make_agent(test_db, tmp_path, llm=Spy(always_fault=drop_opt_out)).run("A-FIBER")
    assert seen == [0, 1, 2]
