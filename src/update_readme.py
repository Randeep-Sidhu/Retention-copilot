"""Fill the generated sections of README.md from the reports, so the numbers in the README can never drift from
the numbers in the results.   Run:  python -m src.update_readme

Sections are delimited by  <!-- NAME:START -->  and  <!-- NAME:END -->  (MODEL, RESULTS, SCREENSHOTS).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from .config import REPORT_DIR, ROOT

ORDER = ["none", "none+verify", "full", "full+verify", "rag", "rag+verify"]


def _pct(d: dict) -> str:
    lo, hi = d["ci"]
    return f"{d['k']}/{d['n']} ({100 * d['rate']:.0f}%, CI {100 * lo:.0f}-{100 * hi:.0f})"


def _p(p: float) -> str:
    text = "p < 0.0001" if p < 0.0001 else f"p = {p:.4f}"
    return text if p < 0.05 else f"{text}, not significant at the 5% level"


def _load(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError):
        return None


def render_model(m: dict | None) -> str:
    if not m:
        return "_Run `python -m src.phase1` and then `python -m src.update_readme` to fill this in._"
    lr, gb, d = m["logreg"], m["hist_gb"], m["delta_auc_gb_minus_lr"]
    includes_zero = d["lo"] <= 0 <= d["hi"]
    verdict = ("The interval includes zero, so I kept the logistic model, which I can explain customer by customer."
               if includes_zero else "The interval excludes zero, so gradient boosting is used for the probabilities.")
    pol = {p["policy"]: p for p in m["policies"]}
    proposed = next(p for name, p in pol.items() if name.startswith("Expected"))
    top = next(p for name, p in pol.items() if name.startswith("Top"))
    everyone = pol["Contact everyone"]
    lo, hi = m["sensitivity_net_value_range_per_1000"]
    return (
        f"On the held-out test set ({m['n_test']:,} customers, {m['churn_rate']:.1%} churn overall) the logistic model reaches "
        f"ROC-AUC {lr['roc_auc']:.3f} and PR-AUC {lr['pr_auc']:.3f}. Gradient boosting, given its natural inputs, reaches "
        f"{gb['roc_auc']:.3f} and {gb['pr_auc']:.3f}. The paired bootstrap difference in ROC-AUC is {d['mean']:+.4f} "
        f"(95% CI {d['lo']:+.4f} to {d['hi']:+.4f}). {verdict}\n\n"
        f"Under the assumptions listed in [the model card](reports/model_card.md), contacting only customers whose expected value is "
        f"positive reaches {proposed['contacted_pct']:.0f}% of them and is worth about ${proposed['net_value_per_1000']:,.0f} per 1,000 "
        f"customers. Contacting everyone is worth ${everyone['net_value_per_1000']:,.0f} and a fixed top-20% list ${top['net_value_per_1000']:,.0f}. "
        f"Across the sensitivity grid the proposed policy ranges from ${lo:,.0f} to ${hi:,.0f} per 1,000, so the method and its "
        f"sensitivity are the result here, not the single figure.")


def render_results(models: dict[str, dict]) -> str:
    if not models:
        return ("_The final evaluation has not been added yet. Run `python -m src.evaluate --pool final` (see below), "
                "then `python -m src.update_readme`._")
    n = sorted({m["n"] for m in models.values()})
    rows = ["| model | configuration | first draft compliant | final compliant | offer matches playbook | prompt tokens | model time / run |",
            "|---|---|---|---|---|---|---|"]
    for model, m in models.items():
        for cfg in [c for c in ORDER if c in m["configs"]]:
            c = m["configs"][cfg]
            rows.append(f"| {model} | {cfg} | {_pct(c['first_ok'])} | {_pct(c['final_ok'])} | {100 * c['matches_preferred']['rate']:.0f}% | "
                        f"{c['mean_prompt_tokens']:.0f} | {c['mean_llm_latency_s']:.1f}s |")
    out = [f"{', '.join(str(x) for x in n)} held-out customers, the same for every row. CI is a 95% Wilson interval.\n", "\n".join(rows), ""]
    for model, m in models.items():
        c = m["configs"]
        bits = []
        if "rag+verify" in c:
            r = c["rag+verify"]
            bits.append(f"With retrieval and the verifier loop, {r['final_ok']['k']} of {r['final_ok']['n']} final drafts passed every rule; "
                        f"the first drafts passed {r['first_ok']['k']} of {r['first_ok']['n']}, and the offer matched the segment playbook "
                        f"{100 * r['matches_preferred']['rate']:.0f}% of the time.")
        if "none" in c:
            z = c["none"]
            bits.append(f"With no policy text the same model passed {z['first_ok']['k']} of {z['first_ok']['n']} and matched the playbook "
                        f"{100 * z['matches_preferred']['rate']:.0f}% of the time.")
        contrast = next((x for x in m["contrasts"] if x["contrast"] == "offer matches playbook: RAG vs no policy"), None)
        if contrast:
            bits.append(f"Retrieval versus no policy on playbook matching: {contrast['only_first']} customers only with retrieval, "
                        f"{contrast['only_second']} only without (exact McNemar {_p(contrast['p'])}).")
        out.append(f"**{model}.** " + " ".join(bits) + "\n")
    noise = []
    for model, m in models.items():
        c = m["configs"]
        if "rag" in c and "rag+verify" in c:
            noise.append((model, c["rag"]["first_ok"]["k"], c["rag+verify"]["first_ok"]["k"], c["rag"]["first_ok"]["n"]))
    if noise:
        parts = " and ".join(f"{a} and {b} of {n0} for {model}" for model, a, b, n0 in noise)
        out.append("**How repeatable is it?** The `rag` and `rag+verify` first drafts use identical prompts, yet they passed "
                   f"{parts}. Identical prompts do not always give identical answers on a GPU, so I treat differences of a few "
                   "customers as noise.\n")
    return "\n".join(out).rstrip()


def render_screenshots(img_dir: Path, readme_dir: Path) -> str:
    imgs = sorted([p for p in img_dir.glob("*") if p.suffix.lower() in (".png", ".jpg", ".jpeg")]) if img_dir.exists() else []
    if not imgs:
        return "<!-- add screenshots to docs/img and run python -m src.update_readme -->"
    lines = []
    for p in imgs:
        caption = re.sub(r"^\d+[-_ ]*", "", p.stem).replace("-", " ").replace("_", " ").strip().capitalize()
        rel = p.relative_to(readme_dir).as_posix()
        lines.append(f"![{caption}]({rel})\n*{caption}*\n")
    return "\n".join(lines).rstrip()


def replace_block(text: str, name: str, body: str) -> str:
    pattern = re.compile(rf"(<!-- {name}:START -->)(.*?)(<!-- {name}:END -->)", re.S)
    if not pattern.search(text):
        raise ValueError(f"markers for {name} not found in the README")
    return pattern.sub(lambda m: f"{m.group(1)}\n{body}\n{m.group(3)}", text, count=1)


def update(readme: Path = ROOT / "README.md", reports: Path = REPORT_DIR, img_dir: Path = ROOT / "docs" / "img") -> str:
    text = readme.read_text(encoding="utf-8")
    models = {}
    for f in sorted(reports.glob("agent_eval_*.json")):
        data = _load(f)
        if data:
            models[f.stem[len("agent_eval_"):]] = data
    text = replace_block(text, "MODEL", render_model(_load(reports / "metrics.json")))
    text = replace_block(text, "RESULTS", render_results(models))
    text = replace_block(text, "SCREENSHOTS", render_screenshots(img_dir, readme.parent))
    readme.write_text(text, encoding="utf-8")
    return text


if __name__ == "__main__":
    update()
    print("README.md updated from reports/")
