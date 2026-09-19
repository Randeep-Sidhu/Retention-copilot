import json

from src.compare_models import main, table
from src.evaluate import compute_metrics


def synthetic(config, ok_first, ok_final, cid):
    return {"customer_id": cid, "config": config, "status": "queued", "schema_ok_first": True, "first_ok": ok_first,
            "final_ok": ok_final, "grounded_first": True, "matches_preferred": ok_final, "escalated": False,
            "revisions": 0, "llm_calls": 1, "prompt_tokens": 1000, "completion_tokens": 100, "llm_latency_s": 2.0,
            "first_rules": [], "context_recall": None, "first_message": ""}


def make_report(tmp_path, slug, ok_first):
    scored = [synthetic(cfg, i < ok_first, True, f"C{i}") for cfg in ("none", "rag+verify") for i in range(10)]
    (tmp_path / f"agent_eval_{slug}.json").write_text(json.dumps(compute_metrics(scored), default=float))


def test_comparison_table_has_one_row_per_model_and_config(tmp_path):
    make_report(tmp_path, "granite4-micro", 3)
    make_report(tmp_path, "granite4-small-h", 8)
    md = main(tmp_path)
    assert (tmp_path / "model_comparison.md").exists()
    rows = [line for line in table({"a": json.loads((tmp_path / "agent_eval_granite4-micro.json").read_text())}).splitlines()[2:]]
    assert len(rows) == 2 and "3/10" in md and "8/10" in md and "granite4-small-h" in md
