"""Answer-time retrieval: the facts valid at the time a question asks about, packed small."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, timedelta

from graph_memory.communities import Community
from graph_memory.extract import MONTHS
from graph_memory.model import Edge

RELATION_WORDS = {
    "lives_in": {"live", "lives", "living", "lived", "address", "home", "move", "moved", "where"},
    "works_at": {"work", "works", "job", "firm", "company", "employer", "office"},
    "partner": {"partner", "wife", "husband", "married"},
}
BROAD = re.compile(r"\b(tell me about|summar|overview|everything|what do you know)", re.IGNORECASE)


@dataclass
class Recall:
    as_of: date
    edges: list[Edge]
    communities: list[Community]
    context: str
    epoch: int = 0  # Memory.epoch when this was built; a deletion since makes it stale


def as_of(query: str, now: date) -> date:
    """The valid time a question is about: 'last year', 'in March 2025', 'in 2024', or now."""
    q = query.lower()
    if "last year" in q:
        return now - timedelta(days=365)
    m = re.search(r"\bin (" + "|".join(MONTHS) + r")(?: (\d{4}))?\b", q)
    if m:
        year = int(m.group(2)) if m.group(2) else now.year
        return date(year, MONTHS[m.group(1)], 15)
    m = re.search(r"\bin (\d{4})\b", q)
    if m:
        return date(int(m.group(1)), 7, 1)
    return now


def _tokens(text: str) -> int:
    return int(len(text.split()) * 1.3) + 1


def relevance(edge: Edge, query: str, names: dict[str, str]) -> float:
    words = set(re.findall(r"[a-z]+", query.lower()))
    score = len(words & RELATION_WORDS.get(edge.relation, set(edge.relation.split("_"))))
    score += 2 * len(words & set(names.get(edge.object, "").lower().split()))
    return score


def line(edge: Edge, names: dict[str, str]) -> str:
    until = edge.valid_to.isoformat() if edge.valid_to else "now"
    since = edge.valid_from.isoformat() if edge.valid_from else "unknown"
    return (f"- {names.get(edge.subject, edge.subject)} {edge.relation.replace('_', ' ')} "
            f"{names.get(edge.object, edge.object)} (from {since} to {until}; learned {edge.recorded_at})")


def pack(edges: list[Edge], communities: list[Community], query: str, now: date,
         names: dict[str, str], budget: int = 4096) -> Recall:
    when = as_of(query, now)
    valid = [e for e in edges if e.current and e.valid_at(when)]
    valid.sort(key=lambda e: (-relevance(e, query, names), e.relation))
    history = [e for e in edges if e.current and not e.valid_at(when)
               and relevance(e, query, names) > 0]
    history.sort(key=lambda e: e.valid_from or date.min, reverse=True)

    sections: list[str] = [f"Facts valid on {when.isoformat()}:"] + [line(e, names) for e in valid]
    summaries = [f"- {c.summary}" for c in communities if c.summary]
    if summaries:
        block = ["Related groups:"] + summaries
        sections = block + sections if BROAD.search(query) else sections + block
    if history:
        sections += ["Earlier or later facts:"] + [line(e, names) for e in history]

    out, used = [], 0
    for s in sections:
        cost = _tokens(s)
        if used + cost > budget:
            break
        out.append(s)
        used += cost
    return Recall(when, valid, communities, "\n".join(out))
