"""The answer loop: read evidence, then answer with citations, search again, or abstain."""

from __future__ import annotations

import json
import os
import re
import urllib.request
from dataclasses import dataclass, field
from typing import Protocol

from retrieval_kit.cite import Citation, find_span
from retrieval_kit.index.bm25 import tokenize
from retrieval_kit.retriever import Hit, Retriever


@dataclass
class Claim:
    sentence: str
    passage: int  # index into the evidence list shown to the model
    quote: str


@dataclass
class Draft:
    """What the model decided this round. Exactly one of the three is set."""

    claims: list[Claim] = field(default_factory=list)
    follow_up: str | None = None
    abstain: bool = False


class Model(Protocol):
    def draft(self, question: str, evidence: list[str]) -> Draft: ...


@dataclass
class Answer:
    text: str
    citations: list[Citation]
    abstained: bool
    rounds: int
    queries: list[str]


class ExtractiveModel:
    """Answers with the evidence sentence that best covers the question. Runs without an LLM."""

    _SENT = re.compile(r"(?:[^.!?\n]|[.!?](?=\S))+[.!?]?")

    def draft(self, question: str, evidence: list[str]) -> Draft:
        q = set(tokenize(question))
        best: tuple[float, int, str] | None = None
        for i, passage in enumerate(evidence):
            for m in self._SENT.finditer(passage):
                sent = m.group().strip()
                overlap = len(q & set(tokenize(sent))) / max(len(q), 1)
                if sent and (best is None or overlap > best[0]):
                    best = (overlap, i, sent)
        if best is None or best[0] < 0.3:
            return Draft(abstain=True)
        return Draft(claims=[Claim(best[2], best[1], best[2])])


class OpenAIChatModel:
    """Any OpenAI-compatible chat endpoint (vLLM, SGLang, a hosted API)."""

    PROMPT = (
        "Answer the question using only the numbered passages. Reply with JSON, one of:\n"
        '{"claims": [{"sentence": str, "passage": int, "quote": str}]}  quote is copied verbatim\n'
        '{"follow_up": str}  a new search query, when the passages are not enough\n'
        '{"abstain": true}  when no passage supports an answer\n'
    )

    def __init__(self, model: str, endpoint: str | None = None, api_key: str | None = None):
        self.model = model
        self.endpoint = (endpoint or os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1")).rstrip("/")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "EMPTY")

    def draft(self, question: str, evidence: list[str]) -> Draft:
        passages = "\n\n".join(f"[{i}] {p}" for i, p in enumerate(evidence))
        body = {
            "model": self.model,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": self.PROMPT},
                {"role": "user", "content": f"Passages:\n{passages}\n\nQuestion: {question}"},
            ],
        }
        req = urllib.request.Request(
            f"{self.endpoint}/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            content = json.load(resp)["choices"][0]["message"]["content"]
        return parse_draft(content)


def parse_draft(content: str) -> Draft:
    data = json.loads(content)
    if data.get("abstain"):
        return Draft(abstain=True)
    if data.get("follow_up"):
        return Draft(follow_up=str(data["follow_up"]))
    return Draft(claims=[Claim(c["sentence"], int(c["passage"]), c["quote"]) for c in data.get("claims", [])])


class Answerer:
    def __init__(
        self,
        retriever: Retriever,
        model: str | Model | None = None,
        max_rounds: int = 3,
        answer_top: int = 8,
        abstain_threshold: float = 0.35,
    ):
        self.retriever = retriever
        if model is None:
            self.model: Model = ExtractiveModel()
        elif isinstance(model, str):
            self.model = OpenAIChatModel(model)
        else:
            self.model = model
        self.max_rounds, self.answer_top, self.abstain_threshold = max_rounds, answer_top, abstain_threshold

    def ask(self, question: str) -> Answer:
        evidence: list[Hit] = []
        seen_pages: set[tuple[str, int]] = set()
        queries = [question]
        rounds = 0
        while rounds < self.max_rounds:
            rounds += 1
            hits = [h for h in self.retriever.search(queries[-1]) if h.score >= self.abstain_threshold]
            new_pages = {(h.chunk.doc, h.chunk.page) for h in hits} - seen_pages
            known = {h.chunk.id for h in evidence}
            evidence += [h for h in hits if h.chunk.id not in known]
            evidence.sort(key=lambda h: -h.score)
            evidence = evidence[: self.answer_top]
            if rounds > 1 and not new_pages:
                break  # a round that adds no new evidence page will not be followed by a better one
            seen_pages |= new_pages
            if not evidence:
                continue
            draft = self.model.draft(question, [h.chunk.text for h in evidence])
            if draft.abstain:
                break
            if draft.follow_up:
                queries.append(draft.follow_up)
                continue
            answer = self._cite(draft, evidence, rounds, queries)
            if answer is not None:
                return answer
        return Answer("I could not find a passage that supports an answer.", [], True, rounds, queries)

    def _cite(self, draft: Draft, evidence: list[Hit], rounds: int, queries: list[str]) -> Answer | None:
        citations, sentences = [], []
        for claim in draft.claims:
            if not 0 <= claim.passage < len(evidence):
                continue
            citation = find_span(evidence[claim.passage].chunk, claim.quote)
            if citation is None:
                continue  # an uncited sentence is dropped, never returned
            citations.append(citation)
            sentences.append(claim.sentence)
        if not citations:
            return None
        return Answer(" ".join(sentences), citations, False, rounds, queries)
