"""Retrieval quality on a hand-written gold set. Run: python -m src.retrieval_eval

Each query has one or more acceptable section ids. hit@k = an acceptable section is in the top k; MRR = mean
reciprocal rank of the first acceptable section. Queries are paraphrases, not copies of section titles.
"""
from __future__ import annotations

import json

from .config import REPORT_DIR
from .retriever import Retriever, load_corpus

GOLD = [
    ("highest discount percentage allowed for a customer with no long-term contract", ["discount_caps#month-to-month-contracts"]),
    ("can we give a price reduction to someone locked in for two years", ["discount_caps#two-year-contracts"]),
    ("discount ceiling for customers on a twelve month plan", ["discount_caps#one-year-contracts"]),
    ("biggest amount in dollars a bill can be reduced", ["discount_caps#absolute-monthly-cap"]),
    ("what can we offer someone who signed up two months ago", ["discount_caps#new-customers-0-3-months", "escalation#new-customers"]),
    ("who qualifies for the free support add-on", ["offer_catalog#tech-support-trial"]),
    ("offer that trades a discount for a longer commitment", ["offer_catalog#term-upgrade"]),
    ("non-promotional call invitation for any customer", ["offer_catalog#service-check-in"]),
    ("customer declined marketing communications", ["consent_and_contact#marketing-consent"]),
    ("how often may we reach out to the same person", ["consent_and_contact#contact-frequency"]),
    ("required wording so people can unsubscribe from texts", ["consent_and_contact#required-opt-out-line"]),
    ("maximum length of a text message", ["consent_and_contact#channel-limits"]),
    ("which words are banned in customer messages", ["message_standards#prohibited-language"]),
    ("is it ok to mention how old the customer is", ["message_standards#protected-characteristics"]),
    ("can the message say the customer is likely to leave", ["message_standards#predictions-and-risk-scores"]),
    ("which numbers may appear in the message", ["message_standards#numbers-must-match-the-offer"]),
    ("best first offer for a monthly fiber customer", ["segment_playbooks#month-to-month-fiber-optic"]),
    ("best first offer for a monthly DSL customer", ["segment_playbooks#month-to-month-dsl"]),
    ("customer only has a phone line and pays month to month", ["segment_playbooks#month-to-month-no-internet-phone-only"]),
    ("one year contract with fiber internet, which offers in order", ["segment_playbooks#one-year-fiber-optic"]),
    ("two year contract on DSL, what offers apply", ["segment_playbooks#two-year-dsl"]),
    ("accounts paying more than one hundred dollars a month", ["escalation#high-value-accounts"]),
    ("when is doing nothing the right decision", ["escalation#when-to-take-no-action"]),
    ("what defines a high risk customer for discount purposes", ["discount_caps#risk-tiers"]),
    ("is a person involved before anything is sent", ["escalation#human-approval"]),
    ("does a service check in need consent", ["consent_and_contact#service-messages"]),
    ("how should a good message be laid out", ["message_standards#structure"]),
    ("example of a well written tech support text", ["message_standards#examples-of-acceptable-messages"]),
]


def evaluate(backend: str) -> dict:
    r = Retriever(load_corpus(), backend=backend)
    n = len(r.chunks)
    ranks, misses = [], []
    for query, expected in GOLD:
        ranked = [c.id for c, _ in r.search(query, k=n)]
        rank = min(ranked.index(e) for e in expected) + 1
        ranks.append(rank)
        if rank > 3:
            misses.append((query, expected[0], rank))
    q = len(GOLD)
    return {"backend": backend, "queries": q,
            "hit@1": sum(x <= 1 for x in ranks) / q, "hit@3": sum(x <= 3 for x in ranks) / q,
            "hit@5": sum(x <= 5 for x in ranks) / q, "mrr": sum(1 / x for x in ranks) / q, "misses_at_3": misses}


def main():
    results = []
    for backend in ("tfidf", "dense", "hybrid"):
        try:
            results.append(evaluate(backend))
        except Exception as exc:                                   # dense needs fastembed + a model download
            print(f"[{backend}] unavailable: {type(exc).__name__}: {str(exc)[:120]}")
    lines = ["| backend | hit@1 | hit@3 | hit@5 | MRR |", "|---|---|---|---|---|"]
    for r in results:
        lines.append(f"| {r['backend']} | {r['hit@1']:.2f} | {r['hit@3']:.2f} | {r['hit@5']:.2f} | {r['mrr']:.2f} |")
    table = "\n".join(lines)
    print(f"\nRetrieval on {results[0]['queries']} gold queries ({len(load_corpus())} policy sections)\n" + table)
    for r in results:
        for query, expected, rank in r["misses_at_3"]:
            print(f"  [{r['backend']}] rank {rank}: '{query}' (wanted {expected})")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "retrieval_eval.md").write_text(
        f"# Retrieval evaluation\n\n{results[0]['queries']} hand-written paraphrase queries over "
        f"{len(load_corpus())} policy sections.\n\n{table}\n", encoding="utf-8")
    (REPORT_DIR / "retrieval_eval.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
