"""ingest -> index + graph -> retrieve -> answer -> verify citations.

* ``retrieval_kit`` indexes the documents and answers with character-span citations.
* ``graph_memory`` remembers what each user has said over time; the facts valid today are
  added to the question so "my employer" means the right company.
* ``cite_check`` re-reads the answer claim by claim against the cited pages and recomputes
  its figures; an answer that fails is revised once, then returned with its remaining issues.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from cite_check import Checker, Issue
from cite_check.retrieve import SourceLibrary
from graph_memory import Memory
from retrieval_kit import Answerer, Citation, Index, Retriever


@dataclass
class GroundedAnswer:
    question: str
    text: str
    citations: list[Citation]
    abstained: bool
    verified: bool
    issues: list[Issue] = field(default_factory=list)
    memory_context: str = ""
    revised: bool = False

    def with_markers(self) -> str:
        return render(self.text, self.citations)


def render(text: str, citations: list[Citation]) -> str:
    """The answer with a ``[doc, p. N]`` marker after it, in the form cite-check reads."""
    if not citations:
        return text
    marks = "".join(sorted({f" [{c.doc}, p. {c.page}]" for c in citations}))
    return text.rstrip() + marks


class KnowledgeBase:
    def __init__(self, folder: str | Path, memory: Memory | None = None, model=None,
                 dense: str | None = None, reranker: str | None = None, chunk_tokens: int = 96,
                 overlap: int = 24, max_rounds: int = 3, checker: Checker | None = None):
        self.folder = Path(folder)
        self.index = Index.build(self.folder, chunk_tokens=chunk_tokens, overlap=overlap)
        self.retriever = Retriever(self.index, dense=dense, reranker=reranker)
        self.answerer = Answerer(self.retriever, model=model, max_rounds=max_rounds)
        self.memory = memory or Memory()
        self.checker = checker or Checker()
        self.sources = SourceLibrary(self.folder, chunk_tokens=chunk_tokens, overlap=overlap, reranker=reranker)

    def remember(self, user: str, text: str, at: str | date | None = None) -> None:
        self.memory.add(user=user, text=text, at=at)

    def _personalise(self, user: str | None, question: str, at) -> tuple[str, str]:
        if user is None:
            return question, ""
        recalled = self.memory.recall(user=user, query=question, at=at)
        names = {e.id: e.name for e in self.memory.graph.entities()}
        employer = next((names[e.object] for e in recalled.edges if e.relation == "works_at"), None)
        place = next((names[e.object] for e in recalled.edges if e.relation == "lives_in"), None)
        q = question
        if employer:
            q = q.replace("my employer", employer).replace("my company", employer).replace("my firm", employer)
        if place:
            q = q.replace("where I live", place)
        return q, recalled.context

    def ask(self, question: str, user: str | None = None, at: str | date | None = None) -> GroundedAnswer:
        query, context = self._personalise(user, question, at)
        result = self.answerer.ask(query)
        answer = GroundedAnswer(question, result.text, result.citations, result.abstained,
                                verified=result.abstained, memory_context=context)
        if result.abstained:
            return answer
        review = self.checker.check(answer.with_markers(), self.sources)
        if review.issues:
            revised = self.checker.revise(review)
            answer.text, answer.revised = revised.report, True
            review = revised
        answer.issues = review.issues
        answer.verified = not review.issues
        return answer
