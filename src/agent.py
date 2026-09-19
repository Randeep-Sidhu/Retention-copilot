"""LangGraph retention agent.

    load_customer (read-only SQL) -> gate (expected value) -> retrieve/policy context -> draft (LLM, structured
    output) -> verify (deterministic policy engine) -> [revise (LLM, violation feedback) -> verify]* -> finalize
    (run log + human review queue)

Run a smoke test:  python -m src.agent --customer auto --config rag+verify --provider ollama
"""
from __future__ import annotations

import argparse
import json
import operator
import re
import time
import warnings
from dataclasses import dataclass, replace
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph

from .config import Economics
from .llm import OllamaLLM, ScriptedLLM
from .policy_engine import Violation, check, review_flags
from .policy_spec import (HIGH_RISK_THRESHOLD, HIGH_VALUE_MONTHLY, MAX_CONTACTS_90D, NEW_CUSTOMER_MAX_TENURE,
                          OFFERS, OPT_OUT_LINE)
from .prompts import build_messages, format_chunks
from .retriever import Retriever, load_corpus
from .schemas import Proposal
from .sql_tool import SqlTool
from .store import RunStore
from .value import expected_value


@dataclass(frozen=True)
class AgentConfig:
    name: str = "rag+verify"
    policy_mode: str = "rag"          # none | full | rag
    verify: bool = True               # False = the verifier only monitors (measures) and never triggers a revision
    max_revisions: int = 2
    retrieval_backend: str = "hybrid" # tfidf | dense | hybrid (falls back to tfidf if dense is unavailable)
    top_k_per_query: int = 1          # sections per single-topic query
    use_gate: bool = True
    system_footer: bool = True        # append the mandatory opt-out line in code if the model left it out
    enqueue: bool = True
    model: str = "granite4:micro"
    temperature: float = 0.0
    num_ctx: int = 8192
    seed: int = 42


CONFIGS = {
    "none": AgentConfig("none", "none", False),
    "none+verify": AgentConfig("none+verify", "none", True),
    "full": AgentConfig("full", "full", False),
    "full+verify": AgentConfig("full+verify", "full", True),
    "rag": AgentConfig("rag", "rag", False),
    "rag+verify": AgentConfig("rag+verify", "rag", True),
}


class AgentState(TypedDict, total=False):
    customer_id: str
    profile: dict
    ev: float
    policy_text: str
    retrieved_ids: list
    proposal: dict | None
    error: str | None
    first_violations: list
    violations: list
    attempts: int
    status: str
    run_id: int | None
    llm_calls: int
    prompt_tokens: int
    completion_tokens: int
    llm_latency_s: float
    trace: Annotated[list, operator.add]


STANDING_QUERIES = [                                   # rules that apply to every message: one topic per query
    "message standards: prohibited language, pressure and absolute promises",
    "message standards: numbers must match the offer",
    "message standards: protected characteristics",
    "message standards: predictions and risk scores",
    "required opt-out line",
    "channel limits sms email characters",
]
_OFFER_ID = re.compile(r"\(([A-Z][A-Z_]{5,})\)")


def segment_query(p: dict) -> str:
    internet = "no internet phone only" if p["internet_service"] == "No" else p["internet_service"]
    return f"{p['contract']}, {internet} preferred order of offers"


def conditional_queries(p: dict) -> list[str]:
    """Extra single-topic queries triggered by the customer's facts (no LLM involved)."""
    qs = [f"discount caps {p['contract']} contracts"]
    if int(p["tenure_months"]) <= NEW_CUSTOMER_MAX_TENURE:
        qs.append("new customers 0-3 months no discounts")
    if float(p["monthly_charges"]) > HIGH_VALUE_MONTHLY:
        qs += ["absolute monthly cap", "high-value accounts priority review"]
    if not int(p["marketing_opt_in"]):
        qs.append("marketing consent: customer has not opted in")
    if int(p["contacts_last_90d"]) >= MAX_CONTACTS_90D:
        qs.append("contact frequency: limit reached, take no action")
    return qs


def retrieve_context(retriever: Retriever, p: dict, k: int = 1):
    """Multi-hop retrieval. Hop 1 fetches the segment playbook; hop 2 fetches the catalog section of every offer the
    playbook names (plus discount risk tiers when a discount is a candidate); standing and conditional single-topic
    queries add the rules that always apply or that this customer's facts trigger."""
    chunks, seen = [], set()

    def add(query, kk=k):
        for chunk, _ in retriever.search(query, k=kk):
            if chunk.id not in seen:
                seen.add(chunk.id); chunks.append(chunk)

    add(segment_query(p))
    offers = list(dict.fromkeys(_OFFER_ID.findall(chunks[0].text))) if chunks else []
    for oid in offers:
        add(f"offer_id: {oid}")
    if {"LOYALTY_DISCOUNT", "TERM_UPGRADE"} & set(offers):
        add("risk tiers: high-risk score and standard tier")
    for q in conditional_queries(p) + STANDING_QUERIES:
        add(q)
    return chunks


def make_retriever(backend: str, chunks) -> Retriever:
    try:
        return Retriever(chunks, backend=backend)
    except Exception as exc:                                    # dense/hybrid need fastembed + a model download
        warnings.warn(f"{backend} retrieval unavailable ({type(exc).__name__}); falling back to tfidf")
        return Retriever(chunks, backend="tfidf")


class RetentionAgent:
    def __init__(self, cfg: AgentConfig = CONFIGS["rag+verify"], llm=None, sql: SqlTool | None = None,
                 store: RunStore | None = None, persist: bool = True, retriever: Retriever | None = None,
                 econ: Economics | None = None, base_url: str | None = None):
        self.cfg, self.econ = cfg, econ or Economics()
        self.sql = sql or SqlTool()
        self.store = store or (RunStore() if persist else None)
        self.chunks = load_corpus()
        self.valid_ids = {c.id for c in self.chunks}
        self.retriever = retriever or (make_retriever(cfg.retrieval_backend, self.chunks) if cfg.policy_mode == "rag" else None)
        self.full_text = format_chunks(self.chunks) if cfg.policy_mode == "full" else ""
        self.llm = llm or OllamaLLM(cfg.model, cfg.temperature, cfg.num_ctx, cfg.seed, base_url)
        self._t0 = time.perf_counter()
        self.graph = self._build()

    # ------------------------------------------------------------------ graph
    def _build(self):
        g = StateGraph(AgentState)
        for name, fn in [("load_customer", self.n_load), ("gate", self.n_gate), ("context", self.n_context),
                         ("draft", self.n_draft), ("verify", self.n_verify), ("revise", self.n_revise),
                         ("finalize", self.n_finalize)]:
            g.add_node(name, fn)
        g.add_edge(START, "load_customer")
        g.add_conditional_edges("load_customer", lambda s: "finalize" if s.get("status") == "error" else "gate",
                                {"finalize": "finalize", "gate": "gate"})
        g.add_conditional_edges("gate", lambda s: "finalize" if s.get("status") == "skipped_gate" else "context",
                                {"finalize": "finalize", "context": "context"})
        g.add_edge("context", "draft")
        g.add_edge("draft", "verify")
        g.add_conditional_edges("verify", self.route_verify, {"revise": "revise", "finalize": "finalize"})
        g.add_edge("revise", "verify")
        g.add_edge("finalize", END)
        return g.compile()

    def _ev(self, node: str, **detail) -> dict:
        return {"node": node, "t": round(time.perf_counter() - self._t0, 3), **detail}

    # ------------------------------------------------------------------ nodes
    def n_load(self, s):
        prof = self.sql.profile(s["customer_id"])
        if prof is None:
            return {"status": "error", "error": "unknown customer_id", "trace": [self._ev("load_customer", error="unknown customer_id")]}
        keys = ("contract", "internet_service", "tenure_months", "monthly_charges", "p_churn")
        return {"profile": prof, "trace": [self._ev("load_customer", **{k: prof[k] for k in keys})]}

    def n_gate(self, s):
        p = s["profile"]
        ev = float(expected_value(float(p["p_churn"]), float(p["monthly_charges"]), self.econ))
        go = ev > 0 or not self.cfg.use_gate
        out = {"ev": ev, "trace": [self._ev("gate", expected_value=round(ev, 2), proceed=go)]}
        if not go:
            out["status"] = "skipped_gate"
        return out

    def n_context(self, s):
        mode, p = self.cfg.policy_mode, s["profile"]
        if mode == "none":
            text, ids, extra = "", [], {}
        elif mode == "full":
            text, ids, extra = self.full_text, sorted(self.valid_ids), {}
        else:
            chunks = retrieve_context(self.retriever, p, k=self.cfg.top_k_per_query)
            text, ids, extra = format_chunks(chunks), [c.id for c in chunks], {"backend": self.retriever.backend}
        return {"policy_text": text, "retrieved_ids": ids,
                "trace": [self._ev("context", mode=mode, sections=len(ids), **extra)]}

    def _with_footer(self, prop):
        """Mandatory disclosures are appended deterministically: they should not depend on a language model."""
        if (self.cfg.system_footer and prop is not None and prop.action == "OFFER" and prop.offer_id in OFFERS
                and OFFERS[prop.offer_id]["promotional"] and OPT_OUT_LINE.lower() not in prop.message.lower()):
            return prop.model_copy(update={"message": f"{prop.message.rstrip()} {OPT_OUT_LINE}.".strip()}), True
        return prop, False

    def _after_llm(self, s, res, node):
        prop, footer = self._with_footer(res.proposal)
        detail = dict(ok=prop is not None, prompt_tokens=res.prompt_tokens, completion_tokens=res.completion_tokens,
                      latency_s=round(res.latency_s, 2), error=res.error, footer_appended=footer)
        if prop is not None:
            detail.update(action=prop.action, offer_id=prop.offer_id, discount_pct=prop.discount_pct,
                          duration_months=prop.duration_months)
        return {"proposal": prop.model_dump() if prop is not None else None, "error": res.error,
                "llm_calls": s["llm_calls"] + 1, "prompt_tokens": s["prompt_tokens"] + res.prompt_tokens,
                "completion_tokens": s["completion_tokens"] + res.completion_tokens,
                "llm_latency_s": s["llm_latency_s"] + res.latency_s, "trace": [self._ev(node, **detail)]}

    def n_draft(self, s):
        msgs = build_messages(self.cfg.policy_mode, s["policy_text"], s["profile"])
        return self._after_llm(s, self.llm.draft(msgs, profile=s["profile"], attempt=0), "draft")

    def n_verify(self, s):
        if s.get("proposal") is None:
            viols = [str(Violation("schema", "the reply was not valid JSON for the required schema"))]
        else:
            found = check(Proposal(**s["proposal"]), s["profile"], self.valid_ids,
                          check_citations=self.cfg.policy_mode != "none")
            viols = [str(v) for v in found]
        out = {"violations": viols, "trace": [self._ev("verify", passed=not viols, violations=viols)]}
        if s["attempts"] == 0:
            out["first_violations"] = viols
        return out

    def route_verify(self, s):
        if self.cfg.verify and s["violations"] and s["attempts"] < self.cfg.max_revisions:
            return "revise"
        return "finalize"

    def n_revise(self, s):
        prev = Proposal(**s["proposal"]).model_dump_json() if s.get("proposal") else None
        msgs = build_messages(self.cfg.policy_mode, s["policy_text"], s["profile"], prev, s["violations"])
        out = self._after_llm(s, self.llm.draft(msgs, profile=s["profile"], attempt=s["attempts"] + 1), "revise")
        out["attempts"] = s["attempts"] + 1
        return out

    def n_finalize(self, s):
        status, prop, viols = s.get("status"), s.get("proposal"), s.get("violations", [])
        if status not in ("error", "skipped_gate"):
            if prop is None:
                status = "failed_schema"
            elif not self.cfg.verify:
                status = "queued_unverified" if prop["action"] == "OFFER" else "no_action"
            elif viols:
                status = "escalated"
            else:
                status = "queued" if prop["action"] == "OFFER" else "no_action"
        run_id = None
        if self.store is not None:
            rec = dict(customer_id=s["customer_id"], config=self.cfg.name, model=getattr(self.llm, "model", "?"),
                       status=status, attempts=s["attempts"], llm_calls=s["llm_calls"],
                       prompt_tokens=s["prompt_tokens"], completion_tokens=s["completion_tokens"],
                       llm_latency_s=s["llm_latency_s"], wall_s=time.perf_counter() - self._t0, proposal=prop,
                       first_violations=s.get("first_violations", []), violations=viols,
                       retrieved_ids=s.get("retrieved_ids", []), trace=s.get("trace", []))
            run_id = self.store.save_run(rec)
            if self.cfg.enqueue and prop is not None and prop["action"] == "OFFER" and status in ("queued", "escalated", "queued_unverified"):
                flags = review_flags(s["profile"], Proposal(**prop)) + (["verifier_failed"] if status == "escalated" else [])
                self.store.enqueue(run_id, s["customer_id"], prop, flags, viols)
        return {"status": status, "run_id": run_id, "trace": [self._ev("finalize", status=status)]}

    # ------------------------------------------------------------------ public API
    def run(self, customer_id: str) -> dict:
        self._t0 = time.perf_counter()
        init = {"customer_id": customer_id, "attempts": 0, "llm_calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
                "llm_latency_s": 0.0, "first_violations": [], "violations": [], "retrieved_ids": [], "trace": []}
        out = dict(self.graph.invoke(init))
        out["config"], out["wall_s"] = self.cfg.name, time.perf_counter() - self._t0
        return out


# ---------------------------------------------------------------------- CLI
def _print_run(out: dict) -> None:
    print(f"\n=== {out['customer_id']} | config {out['config']} | status {out['status']} | "
          f"{out['llm_calls']} LLM call(s), {out['prompt_tokens']}+{out['completion_tokens']} tokens, {out['wall_s']:.1f}s ===")
    for e in out["trace"]:
        detail = {k: v for k, v in e.items() if k not in ("node", "t")}
        print(f"  {e['t']:>6.2f}s  {e['node']:<14} {json.dumps(detail, default=str)[:220]}")
    prop = out.get("proposal")
    if prop:
        print(f"\n  action {prop['action']} | offer {prop['offer_id']} | {prop['discount_pct']}% x {prop['duration_months']} months")
        print(f"  message: {prop['message']}")
        print(f"  rationale: {prop['rationale']}\n  citations: {prop['citations']}")
    if out.get("first_violations"):
        print(f"  first-draft violations: {out['first_violations']}")
    if out.get("violations"):
        print(f"  FINAL violations: {out['violations']}")


def main():
    from .scenarios import sample_test_customers
    ap = argparse.ArgumentParser()
    ap.add_argument("--customer", default="auto", help="customer id, or 'auto' to sample held-out customers")
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--config", default="rag+verify", choices=list(CONFIGS))
    ap.add_argument("--provider", default="ollama", choices=["ollama", "scripted"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--no-persist", action="store_true")
    a = ap.parse_args()
    cfg = CONFIGS[a.config] if not a.model else replace(CONFIGS[a.config], model=a.model)
    llm = ScriptedLLM() if a.provider == "scripted" else None
    agent = RetentionAgent(cfg, llm=llm, persist=not a.no_persist, base_url=a.base_url)
    ids = sample_test_customers(a.n) if a.customer == "auto" else [a.customer]
    for cid in ids:
        _print_run(agent.run(cid))


if __name__ == "__main__":
    main()
