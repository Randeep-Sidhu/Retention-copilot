"""LLM back-ends behind one tiny interface: draft(messages, profile) -> DraftResult."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

from .reference_drafter import reference_proposal
from .schemas import Proposal


@dataclass
class DraftResult:
    proposal: Proposal | None
    raw: str
    error: str | None
    prompt_tokens: int
    completion_tokens: int
    latency_s: float


class OllamaLLM:
    """Granite (or any Ollama model) through LangChain's ChatOllama with schema-constrained output.

    Attempt 0 (the first draft) is deterministic (temperature 0). Revision attempts sample a little (0.3, 0.6) with a
    different seed: at temperature 0 an unchanged prompt gives an unchanged answer, so a model that ignored the
    feedback once would ignore it again.
    """

    def __init__(self, model="granite4:micro", temperature=0.0, num_ctx=8192, seed=42, base_url=None,
                 num_predict=700, timeout=180):
        self.model, self._chains = model, {}
        self._kw = dict(model=model, num_ctx=num_ctx, num_predict=num_predict, client_kwargs={"timeout": timeout})
        if base_url:
            self._kw["base_url"] = base_url
        self._temperature, self._seed = temperature, seed

    def _chain(self, attempt: int):
        if attempt not in self._chains:
            from langchain_ollama import ChatOllama          # lazy: only needed when a real model is used
            temp = self._temperature if attempt == 0 else min(0.9, self._temperature + 0.3 * attempt)
            chat = ChatOllama(temperature=temp, seed=self._seed + attempt, **self._kw)
            self._chains[attempt] = chat.with_structured_output(Proposal, method="json_schema", include_raw=True)
        return self._chains[attempt]

    def draft(self, messages, profile=None, attempt: int = 0) -> DraftResult:
        t0 = time.perf_counter()
        out = self._chain(attempt).invoke(messages)
        raw = out["raw"]
        meta = getattr(raw, "response_metadata", None) or {}
        err = out.get("parsing_error")
        return DraftResult(proposal=out.get("parsed"), raw=str(raw.content), error=str(err)[:300] if err else None,
                           prompt_tokens=int(meta.get("prompt_eval_count", 0) or 0),
                           completion_tokens=int(meta.get("eval_count", 0) or 0),
                           latency_s=time.perf_counter() - t0)


class ScriptedLLM:
    """Offline stand-in for tests, CI and demos: the policy-perfect reference draft, optionally corrupted.

    faults: one entry per call; a callable Proposal -> Proposal corrupts that call's draft, None leaves it clean.
    Calls beyond the list are clean unless always_fault is set.
    """

    def __init__(self, faults: list[Callable[[Proposal], Proposal] | None] | None = None,
                 always_fault: Callable[[Proposal], Proposal] | None = None):
        self.faults, self.always_fault, self.calls = list(faults or []), always_fault, 0
        self.model = "scripted"

    def draft(self, messages, profile=None, attempt: int = 0) -> DraftResult:
        t0 = time.perf_counter()
        prop = reference_proposal(profile)
        fault = self.faults[self.calls] if self.calls < len(self.faults) else self.always_fault
        if fault:
            prop = fault(prop)
        self.calls += 1
        raw = prop.model_dump_json()
        return DraftResult(prop, raw, None, sum(len(m[1]) for m in messages) // 4, len(raw) // 4,
                           time.perf_counter() - t0)
