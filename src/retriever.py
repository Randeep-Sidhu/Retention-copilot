"""Policy retrieval: section-level chunks, TF-IDF / dense (fastembed) / hybrid (reciprocal rank fusion)."""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .policy_docs import POLICY_DIR
from .policy_spec import slug


@dataclass(frozen=True)
class Chunk:
    id: str          # e.g. "discount_caps#month-to-month-contracts"
    doc: str
    section: str
    text: str

    def for_index(self) -> str:
        return f"{self.doc.replace('_', ' ')} - {self.section}. {self.text}"


def load_corpus(policy_dir=POLICY_DIR) -> list[Chunk]:
    """One chunk per '## ' section of every policy document."""
    chunks = []
    for path in sorted(policy_dir.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        for block in re.split(r"(?m)^## ", text)[1:]:
            title, _, body = block.partition("\n")
            chunks.append(Chunk(f"{path.stem}#{slug(title)}", path.stem, title.strip(), body.strip()))
    return chunks


class _Tfidf:
    def __init__(self, texts):
        self.vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
        self.mat = self.vec.fit_transform(texts)

    def scores(self, query: str) -> np.ndarray:
        return (self.mat @ self.vec.transform([query]).T).toarray().ravel()


class _Dense:
    def __init__(self, texts, model="BAAI/bge-small-en-v1.5"):
        from fastembed import TextEmbedding            # imported lazily: optional dependency
        self.model = TextEmbedding(model_name=model)
        mat = np.array(list(self.model.passage_embed(texts)), dtype=float)
        self.mat = mat / np.linalg.norm(mat, axis=1, keepdims=True)

    def scores(self, query: str) -> np.ndarray:
        q = np.array(list(self.model.query_embed(query))[0], dtype=float)
        return self.mat @ (q / np.linalg.norm(q))


class Retriever:
    """backend: 'tfidf' | 'dense' | 'hybrid'. Dense needs `pip install fastembed` and a one-off model download."""

    def __init__(self, chunks: list[Chunk] | None = None, backend: str = "tfidf"):
        self.chunks = chunks or load_corpus()
        self.backend = backend
        texts = [c.for_index() for c in self.chunks]
        self._tfidf = _Tfidf(texts) if backend in ("tfidf", "hybrid") else None
        self._dense = _Dense(texts) if backend in ("dense", "hybrid") else None

    def _ranking(self, scores: np.ndarray) -> np.ndarray:
        return np.argsort(-scores, kind="stable")

    def search(self, query: str, k: int = 4) -> list[tuple[Chunk, float]]:
        if self.backend == "tfidf":
            s = self._tfidf.scores(query)
            order = self._ranking(s)[:k]
            return [(self.chunks[i], float(s[i])) for i in order]
        if self.backend == "dense":
            s = self._dense.scores(query)
            order = self._ranking(s)[:k]
            return [(self.chunks[i], float(s[i])) for i in order]
        fused = np.zeros(len(self.chunks))                       # reciprocal rank fusion
        for s in (self._tfidf.scores(query), self._dense.scores(query)):
            for rank, i in enumerate(self._ranking(s)):
                fused[i] += 1.0 / (60 + rank + 1)
        order = self._ranking(fused)[:k]
        return [(self.chunks[i], float(fused[i])) for i in order]

    def by_id(self, chunk_id: str) -> Chunk:
        return next(c for c in self.chunks if c.id == chunk_id)

    def ids(self) -> set[str]:
        return {c.id for c in self.chunks}
