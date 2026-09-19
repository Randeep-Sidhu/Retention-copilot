"""Evaluation of the agent: 6 configurations x N held-out scenarios, with paired statistics.

Configurations: policy source (none | full | rag) x verifier loop (off | on). With the verifier off, the policy
engine still runs as a passive monitor so every run is scored the same way.

    python -m src.evaluate --model granite4:micro --n 48        # run (resumable) + report
    python -m src.evaluate --report-only                        # recompute the report from stored runs
"""
from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import replace
from pathlib import Path

import matplotlib
import numpy as np
from scipy.stats import binomtest

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .agent import CONFIGS, RetentionAgent  # noqa: E402
from .config import DB_PATH, FIG_DIR, REPORT_DIR, ROOT  # noqa: E402
from .llm import ScriptedLLM  # noqa: E402
from .policy_engine import preferred_offer, required_sections  # noqa: E402
from .scenarios import build_scenarios, scenario_mix  # noqa: E402
from .sql_tool import SqlTool  # noqa: E402
from .store import RunStore  # noqa: E402

EVAL_DB = ROOT / "db" / "eval_runs.db"
ORDER = ["none", "none+verify", "full", "full+verify", "rag", "rag+verify"]
_RULE = re.compile(r"^\[(\w+)\]")


def rules_of(violations) -> list[str]:
    return [m.group(1) for v in violations if (m := _RULE.match(v))]


def substantive(violations) -> list[str]:
    """Rule ids of a run's violations, ignoring citation rules (those are reported as 'grounding')."""
    return [r for r in rules_of(violations) if r != "citations"]


def wilson(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    ci = binomtest(k, n).proportion_ci(confidence_level=0.95, method="wilson")
    return float(ci.low), float(ci.high)


def mcnemar_exact(a: list[bool], b: list[bool]) -> tuple[int, int, float]:
    """Paired exact test. Returns (only_a, only_b, p): scenarios where only A / only B succeeded."""
    only_a = sum(x and not y for x, y in zip(a, b))
    only_b = sum(y and not x for x, y in zip(a, b))
    p = 1.0 if only_a + only_b == 0 else float(binomtest(only_a, only_a + only_b, 0.5).pvalue)
    return only_a, only_b, p


# ----------------------------------------------------------------------------- running
def run_eval(scenario_ids, configs, model="granite4:micro", provider="ollama", base_url=None, db_path=DB_PATH,
             runs_db=EVAL_DB, retrieval_backend=None, log=print):
    store = RunStore(runs_db)
    done = {(r["config"], r["model"], r["customer_id"]) for r in store.runs()}
    for name in configs:
        cfg = replace(CONFIGS[name], model=model, enqueue=False)
        if retrieval_backend:
            cfg = replace(cfg, retrieval_backend=retrieval_backend)
        llm = ScriptedLLM() if provider == "scripted" else None
        model_tag = "scripted" if provider == "scripted" else model
        agent = RetentionAgent(cfg, llm=llm, sql=SqlTool(db_path), store=store, base_url=base_url)
        todo = [c for c in scenario_ids if (name, model_tag, c) not in done]
        t0 = time.perf_counter()
        for i, cid in enumerate(todo, 1):
            out = agent.run(cid)
            eta = (time.perf_counter() - t0) / i * (len(todo) - i)
            log(f"[{name}] {i}/{len(todo)} {cid} status={out['status']} calls={out['llm_calls']} "
                f"{out['wall_s']:.1f}s (eta {eta / 60:.1f} min)")
    return store


# ----------------------------------------------------------------------------- metrics
def load_runs(store: RunStore, model: str | None = None) -> list[dict]:
    runs = []
    for r in store.runs():
        if model and r["model"] != model:
            continue
        for key in ("proposal", "first_violations", "violations", "retrieved_ids"):
            r[key] = json.loads(r.pop(f"{key}_json"))
        runs.append(r)
    return runs


def score_run(run: dict, profile: dict) -> dict:
    fv, v, prop = run["first_violations"], run["violations"], run["proposal"]
    chosen = None if prop is None else (prop["offer_id"] if prop["action"] == "OFFER" else "NONE")
    need = required_sections(profile)
    return {
        "customer_id": run["customer_id"], "config": run["config"], "status": run["status"],
        "schema_ok_first": "schema" not in rules_of(fv),
        "first_ok": not substantive(fv), "final_ok": not substantive(v),
        "grounded_first": "citations" not in rules_of(fv),
        "matches_preferred": chosen == preferred_offer(profile),
        "escalated": run["status"] == "escalated", "revisions": run["attempts"], "llm_calls": run["llm_calls"],
        "prompt_tokens": run["prompt_tokens"], "completion_tokens": run["completion_tokens"],
        "llm_latency_s": run["llm_latency_s"], "first_rules": sorted(set(substantive(fv))),
        "context_recall": (len(need & set(run["retrieved_ids"])) / len(need)) if run["retrieved_ids"] else None,
        "first_message": (prop or {}).get("message", ""),
    }


def compute_metrics(scored: list[dict]) -> dict:
    """Aggregate per configuration over the scenarios that every configuration completed (paired design)."""
    by_cfg: dict[str, dict[str, dict]] = {}
    for s in scored:
        by_cfg.setdefault(s["config"], {})[s["customer_id"]] = s
    common = sorted(set.intersection(*[set(v) for v in by_cfg.values()])) if by_cfg else []
    out = {"n": len(common), "configs": {}, "contrasts": [], "rule_rates": {}, "scenarios": common}
    for name in [c for c in ORDER if c in by_cfg]:
        rows = [by_cfg[name][c] for c in common]
        n = len(rows)
        def rate(key, rows=rows, n=n):
            k = sum(bool(r[key]) for r in rows)
            return {"k": k, "n": n, "rate": k / n if n else float("nan"), "ci": wilson(k, n)}
        grounded = [r for r in rows if not name.startswith("none")]
        recalls = [r["context_recall"] for r in rows if r["context_recall"] is not None and name.startswith("rag")]
        out["configs"][name] = {
            "first_ok": rate("first_ok"), "final_ok": rate("final_ok"), "escalated": rate("escalated"),
            "schema_ok_first": rate("schema_ok_first"), "matches_preferred": rate("matches_preferred"),
            "grounded_first": (rate("grounded_first") if grounded else None),
            "mean_llm_calls": float(np.mean([r["llm_calls"] for r in rows])),
            "mean_revisions": float(np.mean([r["revisions"] for r in rows])),
            "mean_prompt_tokens": float(np.mean([r["prompt_tokens"] for r in rows])),
            "mean_completion_tokens": float(np.mean([r["completion_tokens"] for r in rows])),
            "mean_llm_latency_s": float(np.mean([r["llm_latency_s"] for r in rows])),
            "mean_context_recall": float(np.mean(recalls)) if recalls else None,
        }
        counts: dict[str, int] = {}
        for r in rows:
            for rule in r["first_rules"]:
                counts[rule] = counts.get(rule, 0) + 1
        out["rule_rates"][name] = {k: v / n for k, v in sorted(counts.items(), key=lambda kv: -kv[1])}

    def contrast(label, a_cfg, a_key, b_cfg, b_key):
        if a_cfg in by_cfg and b_cfg in by_cfg:
            a = [by_cfg[a_cfg][c][a_key] for c in common]
            b = [by_cfg[b_cfg][c][b_key] for c in common]
            only_a, only_b, p = mcnemar_exact(a, b)
            out["contrasts"].append({"contrast": label, "only_first": only_a, "only_second": only_b, "p": p})

    contrast("first draft: full policy vs no policy", "full", "first_ok", "none", "first_ok")
    contrast("first draft: RAG vs no policy", "rag", "first_ok", "none", "first_ok")
    contrast("first draft: RAG vs full policy", "rag", "first_ok", "full", "first_ok")
    for base in ("none", "full", "rag"):
        contrast(f"verifier loop on {base}: final vs first draft", f"{base}+verify", "final_ok", f"{base}+verify", "first_ok")
    return out


# ----------------------------------------------------------------------------- report
def pct(d):
    return f"{d['k']}/{d['n']} ({100 * d['rate']:.0f}%; {100 * d['ci'][0]:.0f}-{100 * d['ci'][1]:.0f})"


def make_figure(metrics: dict, path: Path):
    names = list(metrics["configs"])
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    for off, key, label, color in [(-0.2, "first_ok", "first draft", "#8fb3d9"), (0.2, "final_ok", "after verifier loop / final", "#1f4e79")]:
        rates = np.array([metrics["configs"][n][key]["rate"] for n in names]) * 100
        lo = np.array([metrics["configs"][n][key]["ci"][0] for n in names]) * 100
        hi = np.array([metrics["configs"][n][key]["ci"][1] for n in names]) * 100
        ax.bar(x + off, rates, 0.38, yerr=[rates - lo, hi - rates], capsize=3, label=label, color=color)
    ax.set_xticks(x); ax.set_xticklabels(names); ax.set_ylim(0, 105)
    ax.set_ylabel("Policy-compliant drafts (%)")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)
    ax.set_title(f"Compliance by configuration (n={metrics['n']} held-out scenarios, 95% Wilson CI)")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def write_report(metrics: dict, model: str, mix: dict, examples: list[str], out_dir: Path = REPORT_DIR) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    rows = ["| config | first draft compliant | final compliant | escalated | offer matches playbook | prompt tokens | LLM calls | LLM time / run |",
            "|---|---|---|---|---|---|---|---|"]
    for name, m in metrics["configs"].items():
        rows.append(f"| {name} | {pct(m['first_ok'])} | {pct(m['final_ok'])} | {m['escalated']['k']} | "
                    f"{100 * m['matches_preferred']['rate']:.0f}% | {m['mean_prompt_tokens']:.0f} | "
                    f"{m['mean_llm_calls']:.2f} | {m['mean_llm_latency_s']:.1f}s |")
    con = ["| contrast (paired, same scenarios) | only first succeeds | only second succeeds | exact McNemar p |", "|---|---|---|---|"]
    con += [f"| {c['contrast']} | {c['only_first']} | {c['only_second']} | {c['p']:.4f} |" for c in metrics["contrasts"]]
    all_rules = sorted({r for v in metrics["rule_rates"].values() for r in v})
    rules = ["| first-draft violation (share of scenarios) | " + " | ".join(metrics["configs"]) + " |",
             "|---|" + "---|" * len(metrics["configs"])]
    rules += [f"| {r} | " + " | ".join(f"{100 * metrics['rule_rates'][c].get(r, 0):.0f}%" for c in metrics["configs"]) + " |"
              for r in all_rules]
    ground = [f"- {n}: valid citations on {pct(m['grounded_first'])} of first drafts" for n, m in metrics["configs"].items() if m["grounded_first"]]
    recall = [f"- {n}: mean context recall {m['mean_context_recall']:.2f}" for n, m in metrics["configs"].items() if m["mean_context_recall"] is not None]
    md = [f"# Agent evaluation\n\nModel: `{model}` (temperature 0). {metrics['n']} held-out customers who pass the expected-value gate, "
          "stratified to over-represent edge cases (no marketing consent, contact limit reached, new customers, high risk, high value). "
          "Every configuration sees the same customers.\n",
          "**Scenario mix** (a customer can carry several tags): " + ", ".join(f"{k} {v}" for k, v in mix.items()) + "\n",
          "*Compliant* = no policy violation from the deterministic verifier, ignoring citation rules. Compliance of a verifier-loop "
          "configuration's **final** draft is high by construction, which is why first-draft compliance (the effect of the prompt and "
          "the retrieved policy) is reported separately. 'Offer matches playbook' compares the final offer with the first eligible "
          "entry in the segment's preferred order. CI = 95% Wilson interval.\n",
          "## Results\n" + "\n".join(rows), "\n## Paired comparisons\n" + "\n".join(con),
          "\n## Which rules the first drafts break\n" + "\n".join(rules),
          "\n## Grounding and retrieval\n" + "\n".join(ground + recall),
          "\n![compliance](figures/agent_eval.png)\n"]
    if examples:
        md.append("## Example repairs (first draft -> verifier feedback -> final)\n" + "\n\n".join(examples))
    (out_dir / "agent_eval.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (out_dir / "agent_eval.json").write_text(json.dumps(metrics, indent=2, default=float), encoding="utf-8")
    make_figure(metrics, out_dir / "figures" / "agent_eval.png")


def repair_examples(runs: list[dict], limit: int = 3) -> list[str]:
    ex = []
    for r in runs:
        if r["config"].endswith("+verify") and r["first_violations"] and not r["violations"] and r["proposal"]:
            ex.append(f"**{r['customer_id']}** ({r['config']})\n- first-draft violations: " + "; ".join(r["first_violations"])[:400]
                      + f"\n- final message: {r['proposal']['message']}")
        if len(ex) >= limit:
            break
    return ex


def report(store: RunStore, model: str, db_path=DB_PATH, out_dir: Path = REPORT_DIR) -> dict:
    runs = load_runs(store, model)
    sql = SqlTool(db_path)
    profiles = {c: sql.profile(c) for c in {r["customer_id"] for r in runs}}
    scored = [score_run(r, profiles[r["customer_id"]]) for r in runs]
    metrics = compute_metrics(scored)
    write_report(metrics, model, scenario_mix(metrics["scenarios"], db_path), repair_examples(runs), out_dir)
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="granite4:micro")
    ap.add_argument("--provider", default="ollama", choices=["ollama", "scripted"])
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--configs", default="all", help="comma-separated names or 'all'")
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--fresh", action="store_true", help="delete previous eval runs first")
    ap.add_argument("--retrieval", default=None, choices=["tfidf", "dense", "hybrid"])
    a = ap.parse_args()
    if a.fresh and EVAL_DB.exists():
        EVAL_DB.unlink()
    model_tag = "scripted" if a.provider == "scripted" else a.model
    if not a.report_only:
        ids = build_scenarios(a.n)
        names = ORDER if a.configs == "all" else a.configs.split(",")
        run_eval(ids, names, a.model, a.provider, a.base_url, retrieval_backend=a.retrieval)
    m = report(RunStore(EVAL_DB), model_tag)
    print(f"\nReport written to {REPORT_DIR / 'agent_eval.md'} ({m['n']} scenarios x {len(m['configs'])} configs)")
    for name, c in m["configs"].items():
        print(f"  {name:<12} first {pct(c['first_ok']):<26} final {pct(c['final_ok'])}")


if __name__ == "__main__":
    main()
