"""The answer loop: a controller with four tools.

Each round the model reads the evidence so far and takes one action:

* ``search``: a follow-up query through the hybrid retriever;
* ``open_section``: read a whole section by its number ("Article 7"), from the chunks' section paths;
* ``follow_reference``: read the section a passage points at ("as defined in Section 1.1"); a reference
  that no heading carries falls back to a search for it, which is where BM25's exact matching helps;
* ``answer``: claims, each with a verbatim quote from one passage; or it abstains.

Every claim must be located in its passage as a character span. A claim whose quote cannot be found
triggers one more search with the claim itself; claims that still fail are dropped and reported in
``Answer.unanswered``, never returned uncited.
"""

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
    """What the model decided this round. Exactly one is set."""

    claims: list[Claim] = field(default_factory=list)
    follow_up: str | None = None  # the search tool
    abstain: bool = False
    open_section: str | None = None  # "Article 7"
    follow_reference: str | None = None  # "Section 1.1"


class Model(Protocol):
    def draft(self, question: str, evidence: list[str]) -> Draft: ...


@dataclass
class Answer:
    text: str
    citations: list[Citation]
    abstained: bool
    rounds: int
    queries: list[str]
    actions: list[str] = field(default_factory=list)  # the tool used each round, then answer or abstain
    unanswered: list[str] = field(default_factory=list)  # claims dropped: no span supports them


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
        '{"open_section": str}  read a whole section by its number, e.g. "Article 7"\n'
        '{"follow_reference": str}  read the section a passage refers to, e.g. "Section 1.1"\n'
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
    if data.get("open_section"):
        return Draft(open_section=str(data["open_section"]))
    if data.get("follow_reference"):
        return Draft(follow_reference=str(data["follow_reference"]))
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
        queries, actions = [question], ["search"]
        step: tuple[str, str] = ("search", question)
        partial: Answer | None = None
        rounds = 0
        while rounds < self.max_rounds:
            rounds += 1
            tool, arg = step
            if tool != "search" and not (chunks := self.retriever.index.section(arg)):
                tool = "search"  # no heading carries that number: search for the reference itself
                queries.append(arg)
            if tool == "search":
                hits = [h for h in self.retriever.search(arg) if h.score >= self.abstain_threshold]
                new_pages = {(h.chunk.doc, h.chunk.page) for h in hits} - seen_pages
                self._add(evidence, hits)
                if rounds > 1 and not new_pages:
                    break  # a round that adds no new evidence page will not be followed by a better one
                seen_pages |= new_pages
            elif not self._add(evidence, [Hit(c, 1.0) for c in chunks[: self.answer_top]]) and rounds > 1:
                break  # the section is already in view
            if not evidence:
                continue
            draft = self.model.draft(question, [h.chunk.text for h in evidence])
            if draft.abstain:
                break
            tools = {"search": draft.follow_up, "open_section": draft.open_section,
                     "follow_reference": draft.follow_reference}
            if any(tools.values()):
                step = next((tool, arg) for tool, arg in tools.items() if arg)
                actions.append(step[0])
                if draft.follow_up:
                    queries.append(draft.follow_up)
                continue
            answer = self._cite(draft, evidence, rounds, queries, actions)
            if answer is not None and not answer.unanswered:
                answer.actions.append("answer")
                return answer
            if answer is not None:
                partial = answer
            failed = answer.unanswered if answer is not None else [c.sentence for c in draft.claims]
            if not failed:
                break
            step = ("search", failed[0])  # look once more for the claim that no quoted span supports
            queries.append(failed[0])
            actions.append("search")
        if partial is not None:
            partial.rounds, partial.actions = rounds, actions + ["answer"]
            return partial
        return Answer("I could not find a passage that supports an answer.", [], True, rounds, queries,
                      actions + ["abstain"])

    def _add(self, evidence: list[Hit], hits: list[Hit]) -> bool:
        """Add unseen chunks, keep the best ``answer_top`` by score; True when anything new was added."""
        known = {h.chunk.id for h in evidence}
        fresh = [h for h in hits if h.chunk.id not in known]
        evidence += fresh
        evidence.sort(key=lambda h: (-h.score, h.chunk.id))
        del evidence[self.answer_top :]
        return bool(fresh)

    def _cite(self, draft: Draft, evidence: list[Hit], rounds: int, queries: list[str],
              actions: list[str]) -> Answer | None:
        citations, sentences, unanswered = [], [], []
        for claim in draft.claims:
            citation = find_span(evidence[claim.passage].chunk, claim.quote) if 0 <= claim.passage < len(
                evidence) else None
            if citation is None:
                unanswered.append(claim.sentence)  # an uncited sentence is dropped, never returned
                continue
            citations.append(citation)
            sentences.append(claim.sentence)
        if not citations:
            return None
        return Answer(" ".join(sentences), citations, False, rounds, queries, list(actions), unanswered)
