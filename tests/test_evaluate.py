"""Evaluation harness: statistics helpers, metric aggregation, resumable runs and report files (offline)."""
import json

import pytest

from src.evaluate import (compute_metrics, mcnemar_exact, report, rules_of, run_eval, subsample, substantive, wilson)
from src.store import RunStore

IDS = ["A-FIBER", "B-ONEYR", "C-OPTOUT", "D-LIMIT", "F-2YR"]


def test_rule_parsing_ignores_citation_rules_for_substantive_compliance():
    v = ["[discount_cap] too high", "[citations] unknown id", "[numbers] wrong"]
    assert rules_of(v) == ["discount_cap", "citations", "numbers"]
    assert substantive(v) == ["discount_cap", "numbers"]


def test_wilson_interval_brackets_the_rate_and_is_wider_for_small_n():
    lo, hi = wilson(30, 48)
    assert lo < 30 / 48 < hi
    assert (wilson(3, 5)[1] - wilson(3, 5)[0]) > (hi - lo)


def test_mcnemar_exact_matches_hand_calculation():
    only_a, only_b, p = mcnemar_exact([True] * 5 + [True] * 3, [False] * 5 + [True] * 3)
    assert (only_a, only_b) == (5, 0) and p == pytest.approx(0.0625)          # 2 * 0.5**5
    assert mcnemar_exact([True, False], [True, False])[2] == 1.0


def synthetic(config, ok_first, ok_final, cid):
    return {"customer_id": cid, "config": config, "status": "queued", "schema_ok_first": True, "first_ok": ok_first,
            "final_ok": ok_final, "grounded_first": True, "matches_preferred": ok_final, "escalated": not ok_final,
            "revisions": 0 if ok_first else 1, "llm_calls": 1 if ok_first else 2, "prompt_tokens": 1000,
            "completion_tokens": 100, "llm_latency_s": 2.0, "first_rules": [] if ok_first else ["discount_cap"],
            "context_recall": 0.9 if config.startswith("rag") else None, "first_message": ""}


def test_metrics_aggregate_per_config_with_paired_contrasts():
    ids = [f"C{i}" for i in range(10)]
    scored = []
    for i, cid in enumerate(ids):
        scored.append(synthetic("none", i < 2, i < 2, cid))                  # 2/10 compliant without policy
        scored.append(synthetic("rag", i < 8, i < 8, cid))                   # 8/10 with RAG (superset of the 2)
        scored.append(synthetic("rag+verify", i < 8, True, cid))             # verifier repairs the rest
    m = compute_metrics(scored)
    assert m["n"] == 10
    assert m["configs"]["none"]["first_ok"]["k"] == 2 and m["configs"]["rag"]["first_ok"]["k"] == 8
    assert m["configs"]["rag+verify"]["final_ok"]["k"] == 10
    c = {x["contrast"]: x for x in m["contrasts"]}["first draft: RAG vs no policy"]
    assert (c["only_first"], c["only_second"]) == (6, 0) and c["p"] == pytest.approx(2 * 0.5 ** 6)
    assert m["configs"]["rag"]["mean_context_recall"] == pytest.approx(0.9)
    assert m["configs"]["none"]["grounded_first"] is None                    # nothing to cite without a policy
    assert m["rule_rates"]["none"]["discount_cap"] == pytest.approx(0.8)


def test_only_scenarios_completed_by_every_config_are_compared():
    scored = [synthetic("none", True, True, "A"), synthetic("none", True, True, "B"), synthetic("rag", True, True, "A")]
    assert compute_metrics(scored)["scenarios"] == ["A"]


def test_end_to_end_with_scripted_llm_is_resumable_and_writes_the_report(test_db, tmp_path):
    runs_db, out = tmp_path / "eval.db", tmp_path / "reports"
    logs = []
    run_eval(IDS, ["none", "rag+verify"], provider="scripted", db_path=test_db, runs_db=runs_db,
             retrieval_backend="tfidf", log=logs.append)
    assert len(RunStore(runs_db).runs()) == 10 and len(logs) == 10
    run_eval(IDS, ["none", "rag+verify"], provider="scripted", db_path=test_db, runs_db=runs_db,
             retrieval_backend="tfidf", log=logs.append)
    assert len(RunStore(runs_db).runs()) == 10                                 # nothing re-run
    m = report(RunStore(runs_db), "scripted", db_path=test_db, out_dir=out)
    assert m["n"] == 5 and m["configs"]["rag+verify"]["final_ok"]["k"] == 5
    assert all(m["configs"][c]["first_ok"]["k"] == 5 for c in m["configs"])   # the reference drafter is policy-perfect
    assert (out / "agent_eval_scripted.md").exists() and (out / "figures" / "agent_eval_scripted.png").exists()
    assert json.loads((out / "agent_eval_scripted.json").read_text())["n"] == 5


def test_subsample_is_evenly_spaced_unique_and_order_preserving():
    ids = [f"C{i:02d}" for i in range(48)]
    sub = subsample(ids, 8)
    assert sub == ["C00", "C06", "C12", "C18", "C24", "C30", "C36", "C42"]
    assert subsample(ids, 100) == ids and len(set(subsample(ids, 7))) == 7
