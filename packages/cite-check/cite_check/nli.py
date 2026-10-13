"""Support checks for (claim, passage) pairs.

Three tiers, cheapest first: an NLI model screens every pair; pairs it is unsure about go to the
verifier (``hanaml/llama-cite-verify-3b``); pairs where the two disagree go to the reviewer.
Without model weights, ``LexicalNLI`` stands in for the NLI tier so the pipeline runs offline.
"""

from __future__ import annotations

import json
import os
import re
import urllib.request
from dataclasses import dataclass
from typing import Protocol

from cite_check.units import parse, same

STOP = set("the a an of in on at to for and or was were is are be by with from as that this its it than".split())
NEGATION = re.compile(r"\b(not|no|never|without|neither|nor|fell|declined|decreased|lower)\b", re.I)


@dataclass(frozen=True)
class Judgement:
    entail: float
    contradict: float
    tier: str  # nli | verifier | reviewer

    @property
    def label(self) -> str:
        if self.contradict > max(self.entail, 0.5):
            return "contradicted"
        return "supported" if self.entail >= 0.5 else "unsupported"


class SupportModel(Protocol):
    def judge(self, claim: str, passage: str) -> Judgement: ...


def _content(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z][a-z-]+", text.lower()) if w not in STOP}


class LexicalNLI:
    """Coverage of the claim's content words and figures by the passage, with polarity checks."""

    def judge(self, claim: str, passage: str) -> Judgement:
        words = _content(claim)
        coverage = len(words & _content(passage)) / max(len(words), 1)
        claimed, found = parse(claim), parse(passage)
        figures_ok = all(any(same(c, f) for f in found) for c in claimed)
        figure_clash = bool(claimed) and not figures_ok and bool(found)
        polarity_clash = bool(NEGATION.search(claim)) != bool(NEGATION.search(passage)) and coverage > 0.6
        entail = coverage * (1.0 if figures_ok else 0.3)
        contradict = 0.9 if (figure_clash and coverage > 0.5) or polarity_clash else 0.0
        return Judgement(min(entail, 1.0), contradict, "nli")


class TransformersNLI:
    """DeBERTa-v3 MNLI-style models through Hugging Face transformers."""

    def __init__(self, name: str):
        try:
            from transformers import pipeline
        except ImportError as exc:  # pragma: no cover
            raise ImportError('NLI models need: pip install "cite-check[models]"') from exc
        self.pipe = pipeline("text-classification", model=name, top_k=None)

    def judge(self, claim: str, passage: str) -> Judgement:  # pragma: no cover - needs weights
        scores = {d["label"].lower(): d["score"] for d in self.pipe({"text": passage, "text_pair": claim})}
        return Judgement(scores.get("entailment", 0.0), scores.get("contradiction", 0.0), "nli")


class ChatVerifier:
    """A verifier or reviewer behind an OpenAI-compatible endpoint, answering in JSON."""

    def __init__(self, model: str, tier: str, endpoint: str | None = None):
        self.model, self.tier = model, tier
        self.endpoint = (endpoint or os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1")).rstrip("/")

    def judge(self, claim: str, passage: str) -> Judgement:
        prompt = (f"Passage:\n{passage}\n\nClaim:\n{claim}\n\nDoes the passage support the claim? Reply with "
                  'JSON {"label": "supported" | "unsupported" | "contradicted", "confidence": 0-1}.')
        body = {"model": self.model, "temperature": 0, "response_format": {"type": "json_object"},
                "messages": [{"role": "user", "content": prompt}]}
        req = urllib.request.Request(f"{self.endpoint}/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            out = json.loads(json.load(resp)["choices"][0]["message"]["content"])
        conf = float(out.get("confidence", 1.0))
        label = out.get("label", "unsupported")
        return Judgement(conf if label == "supported" else 1 - conf,
                         conf if label == "contradicted" else 0.0, self.tier)


def load(name: str | None, tier: str) -> SupportModel | None:
    if not name:
        return None
    if name == "lexical":
        return LexicalNLI()
    if tier == "nli":
        return TransformersNLI(name)
    return ChatVerifier(name, tier)


class Cascade:
    def __init__(self, nli: SupportModel, verifier: SupportModel | None = None,
                 reviewer: SupportModel | None = None, accept: float = 0.9, reject: float = 0.2):
        self.nli, self.verifier, self.reviewer = nli, verifier, reviewer
        self.accept, self.reject = accept, reject

    def judge(self, claim: str, passage: str) -> Judgement:
        first = self.nli.judge(claim, passage)
        if first.entail >= self.accept or first.contradict >= self.accept or self.verifier is None:
            return first
        if first.entail <= self.reject and first.contradict < 0.5:
            return first
        second = self.verifier.judge(claim, passage)
        if self.reviewer is None or second.label == first.label:
            return second
        return self.reviewer.judge(claim, passage)
