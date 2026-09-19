"""Side-by-side comparison of the final evaluation across models.  Run: python -m src.compare_models"""
from __future__ import annotations

import json
from pathlib import Path

from .config import REPORT_DIR
from .evaluate import ORDER, pct


def load(report_dir: Path = REPORT_DIR, stem: str = "agent_eval") -> dict[str, dict]:
    out = {}
    for f in sorted(report_dir.glob(f"{stem}_*.json")):
        out[f.stem[len(stem) + 1:]] = json.loads(f.read_text(encoding="utf-8"))
    return out


def table(models: dict[str, dict]) -> str:
    rows = ["| model | config | first draft compliant | final compliant | offer matches playbook | prompt tokens | LLM calls | LLM time / run |",
            "|---|---|---|---|---|---|---|---|"]
    for model, m in models.items():
        for cfg in [c for c in ORDER if c in m["configs"]]:
            c = m["configs"][cfg]
            rows.append(f"| {model} | {cfg} | {pct(c['first_ok'])} | {pct(c['final_ok'])} | "
                        f"{100 * c['matches_preferred']['rate']:.0f}% | {c['mean_prompt_tokens']:.0f} | "
                        f"{c['mean_llm_calls']:.2f} | {c['mean_llm_latency_s']:.1f}s |")
    return "\n".join(rows)


def main(report_dir: Path = REPORT_DIR) -> str:
    models = load(report_dir)
    if not models:
        raise SystemExit("No final reports found: run python -m src.evaluate --pool final first")
    n = {m["n"] for m in models.values()}
    md = (f"# Model comparison (final evaluation, {', '.join(str(x) for x in sorted(n))} held-out customers)\n\n"
          "Same customers, prompts and policy for every model. CI = 95% Wilson interval in parentheses.\n\n" + table(models) + "\n")
    (report_dir / "model_comparison.md").write_text(md, encoding="utf-8")
    print(md)
    return md


if __name__ == "__main__":
    main()
