import json

import pytest

from src.evaluate import compute_metrics
from src.update_readme import render_model, render_results, replace_block, update

METRICS = {
    "n_test": 1761, "churn_rate": 0.265,
    "logreg": {"roc_auc": 0.841, "pr_auc": 0.645}, "hist_gb": {"roc_auc": 0.843, "pr_auc": 0.658},
    "delta_auc_gb_minus_lr": {"mean": 0.0024, "lo": -0.0021, "hi": 0.0068},
    "policies": [{"policy": "Contact everyone", "contacted_pct": 100, "net_value_per_1000": 3406},
                 {"policy": "Top 20% by churn risk", "contacted_pct": 20, "net_value_per_1000": 6805},
                 {"policy": "Expected value > 0 (proposed)", "contacted_pct": 40, "net_value_per_1000": 8990}],
}

TEMPLATE = ("# T\n<!-- MODEL:START -->\nold\n<!-- MODEL:END -->\n<!-- RESULTS:START -->\nold\n<!-- RESULTS:END -->\n"
            "<!-- SCREENSHOTS:START -->\nold\n<!-- SCREENSHOTS:END -->\n")


def synthetic(cfg, first, final, cid, match):
    return {"customer_id": cid, "config": cfg, "status": "queued", "schema_ok_first": True, "first_ok": first, "final_ok": final,
            "grounded_first": True, "matches_preferred": match, "escalated": False, "revisions": 0, "llm_calls": 1, "prompt_tokens": 900,
            "completion_tokens": 90, "llm_latency_s": 2.0, "first_rules": [], "context_recall": None, "first_message": ""}


def eval_json(tmp_path, slug):
    scored = []
    for i in range(10):
        scored += [synthetic("none", i < 5, i < 5, f"C{i}", False), synthetic("rag+verify", i < 7, True, f"C{i}", True),
                   synthetic("rag", i < 7, i < 7, f"C{i}", i < 9)]
    (tmp_path / f"agent_eval_{slug}.json").write_text(json.dumps(compute_metrics(scored), default=float))


def test_model_paragraph_states_the_numbers_from_the_report_and_the_verdict():
    METRICS_FULL = {**METRICS, "sensitivity_net_value_range_per_1000": [464, 25851]}
    text = render_model(METRICS_FULL)
    assert "ROC-AUC 0.841" in text and "$8,990" in text and "$464 to $25,851" in text and "includes zero" in text
    other = {**METRICS_FULL, "delta_auc_gb_minus_lr": {"mean": 0.01, "lo": 0.004, "hi": 0.02}}
    assert "excludes zero" in render_model(other)


def test_results_block_has_one_row_per_config_and_reports_the_playbook_contrast(tmp_path):
    eval_json(tmp_path, "granite4-small-h")
    models = {"granite4-small-h": json.loads((tmp_path / "agent_eval_granite4-small-h.json").read_text())}
    text = render_results(models)
    assert text.count("| granite4-small-h |") == 3
    assert "10 of 10 final drafts passed every rule" in text and "first drafts passed 7 of 10" in text
    assert "McNemar p" in text
    assert "How repeatable is it?" in text and "passed 7 and 7 of 10 for granite4-small-h" in text
    assert "exact McNemar p = 0.0039)" in text                       # 9 vs 0 discordant pairs: significant, so no caveat
    from src.update_readme import _p
    assert _p(0.1078) == "p = 0.1078, not significant at the 5% level" and _p(1e-9) == "p < 0.0001"
    assert "has not been added yet" in render_results({})


def test_update_fills_all_blocks_and_is_idempotent(tmp_path):
    (tmp_path / "docs" / "img").mkdir(parents=True)
    (tmp_path / "docs" / "img" / "01-agent-replay.png").write_bytes(b"x")
    readme = tmp_path / "README.md"
    readme.write_text(TEMPLATE)
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "metrics.json").write_text(json.dumps({**METRICS, "sensitivity_net_value_range_per_1000": [1, 2]}))
    eval_json(reports, "granite4-micro")
    first = update(readme, reports, tmp_path / "docs" / "img")
    assert "old" not in first and "ROC-AUC 0.841" in first and "granite4-micro" in first
    assert "![Agent replay](docs/img/01-agent-replay.png)" in first
    assert update(readme, reports, tmp_path / "docs" / "img") == first


def test_missing_markers_are_an_error():
    with pytest.raises(ValueError):
        replace_block("no markers here", "MODEL", "x")
