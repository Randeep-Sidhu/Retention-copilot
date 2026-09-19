"""Retrieval tests (TF-IDF only, so they run offline; the dense backend is evaluated by src.retrieval_eval)."""
from src.retrieval_eval import GOLD, evaluate
from src.retriever import Retriever, load_corpus


def test_corpus_ids_are_unique_and_gold_ids_exist():
    chunks = load_corpus()
    ids = [c.id for c in chunks]
    assert len(ids) == len(set(ids)) >= 30
    for _, expected in GOLD:
        assert all(e in ids for e in expected), expected


def test_keyword_queries_find_their_section():
    r = Retriever(backend="tfidf")
    top3 = lambda q: [c.id for c, _ in r.search(q, k=3)]          # noqa: E731
    assert "consent_and_contact#required-opt-out-line" in top3("required opt-out line for promotional messages")
    assert "segment_playbooks#month-to-month-fiber-optic" in top3("Month-to-month Fiber optic preferred order of offers")
    assert "discount_caps#one-year-contracts" in top3("discount cap one-year contracts")


def test_search_returns_k_distinct_scored_chunks():
    hits = Retriever(backend="tfidf").search("discount cap", k=4)
    assert len(hits) == 4 and len({c.id for c, _ in hits}) == 4
    assert hits[0][1] >= hits[-1][1]


def test_tfidf_baseline_is_better_than_chance():
    res = evaluate("tfidf")
    assert res["hit@5"] > 5 / len(load_corpus())        # chance level for k=5
