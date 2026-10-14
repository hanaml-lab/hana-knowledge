"""The public entry point."""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date

from graph_memory import communities as comm
from graph_memory import notes
from graph_memory.consolidate import Plan, Unit, plan
from graph_memory.embed import BgeM3Embedder, CharNgramEmbedder, Embedder
from graph_memory.extract import Extractor, LLMExtractor, PatternExtractor
from graph_memory.model import Edge, Entity
from graph_memory.recall import Recall, pack
from graph_memory.resolve import FUNCTIONAL, Resolver
from graph_memory.store import GraphStore, InMemoryGraphStore, InMemoryVectorStore, VectorStore


def _date(value: str | date | None) -> date:
    if value is None:
        return date.today()
    return value if isinstance(value, date) else date.fromisoformat(value)


@dataclass
class Episode:
    user: str
    text: str
    when: date
    state: str = "graph"  # graph (consolidated), raw (kept, indexed as text) or cold (forgotten from recall)


_PLANS = re.compile(r"\b(will|plan|plans|prefer|always|never|every|remind|allergic|birthday|going to)\b",
                    re.IGNORECASE)
_FIRST_PERSON = re.compile(r"\b(I|my|me)\b")
# Words with which a user states a one-valued relation about themselves ("I moved to", "I joined").
_RELATION_CUES = {
    "lives_in": {"live", "lives", "living", "lived", "move", "moved", "relocated", "address", "home"},
    "works_at": {"work", "works", "working", "job", "joined", "started", "employer", "company", "firm"},
    "partner": {"partner", "wife", "husband", "married"},
}
_CUES = set().union(*_RELATION_CUES.values())


_STOP = set("the and was were for with you your what where when who how did does have has had this that from "
             "are not but can all any its our out".split())


def _content(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]+", text.lower()) if len(w) > 2 and w not in _STOP}


def prior_access(text: str, age: int, mentions: int) -> float:
    """A hand-set prior for the chance a turn is read again: a first-person statement about where the user
    lives, works or who they are with, a plan or a preference, and mentions of known entities raise it; age
    lowers it. Replace it with a fitted hazard model (``consolidate.p_within`` over its per-step hazards) once
    you log which turns recall uses."""
    words = set(re.findall(r"[a-z]+", text.lower()))
    personal = bool(_FIRST_PERSON.search(text))
    z = (-3.0 + 2.5 * (personal and bool(words & _CUES)) + 1.5 * (personal and bool(_PLANS.search(text)))
         + 1.0 * min(mentions, 2) - 0.05 * age)
    return 1.0 / (1.0 + math.exp(-z))


class Memory:
    """``Memory()`` keeps everything in process. Pass ``neo4j=`` and ``qdrant=`` URLs to persist.

    Every turn passed to ``add`` is kept as an episode, and the graph, the entity vectors and the
    communities are derived from the episodes. ``delete`` removes an episode and rebuilds what was derived
    from it; ``epoch`` counts deletions, and a ``Recall`` carries the epoch it was built in.
    """

    def __init__(
        self,
        neo4j: str | None = None,
        qdrant: str | None = None,
        extractor: str | Extractor = "patterns",
        embedder: str | Embedder = "char-ngram",
        resolve_threshold: float = 0.82,
        community_refresh: str = "on_change",
        recall_tokens: int = 4096,
        graph: GraphStore | None = None,
        vectors: VectorStore | None = None,
        consolidation: str = "eager",
        budget: float = 0.4,
        access: Callable[[str, str, int], float] | None = None,
        conflict: Callable[[str, str], float] | None = None,
    ):
        if graph is None and neo4j:
            from graph_memory.store.neo4j import Neo4jStore
            graph = Neo4jStore(neo4j)
        if vectors is None and qdrant:
            from graph_memory.store.qdrant import QdrantStore
            vectors = QdrantStore(qdrant)
        self.graph = graph or InMemoryGraphStore()
        self.vectors = vectors or InMemoryVectorStore()
        if isinstance(extractor, str):
            extractor = PatternExtractor() if extractor == "patterns" else LLMExtractor(extractor)
        self.extractor = extractor
        if isinstance(embedder, str):
            embedder = CharNgramEmbedder() if embedder == "char-ngram" else BgeM3Embedder(embedder)
        self.resolver = Resolver(self.graph, self.vectors, embedder, resolve_threshold)
        if community_refresh not in {"on_change", "nightly"}:
            raise ValueError("community_refresh is on_change or nightly")
        self.community_refresh = community_refresh
        self.recall_tokens = recall_tokens
        self._communities: list[comm.Community] = []
        self._stale = False
        self._episodes: list[Episode] = []  # every turn, in the order it arrived
        self.epoch = 0
        self._communities_epoch = 0
        if consolidation not in {"eager", "budgeted"}:
            raise ValueError("consolidation is eager or budgeted")
        self.consolidation, self.budget = consolidation, budget
        self.access, self.conflict = access, conflict

    def add(self, user: str, text: str, at: str | date | None = None) -> list[Edge]:
        """Remember a turn. Eager memory writes it to the graph now; budgeted memory keeps it as raw text,
        still recalled, until ``consolidate`` decides whether it earns a graph write."""
        when = _date(at)
        if self.consolidation == "budgeted":
            self._episodes.append(Episode(user, text, when, "raw"))
            return []
        self._episodes.append(Episode(user, text, when))
        return self._derive(user, text, when)

    def consolidate(self, at: str | date | None = None) -> Plan:
        """One budgeted cycle over the raw turns (TIDE): graph-write those that earn it within
        ``budget`` times the cost of writing them all, keep the rest as raw text, and drop old turns nobody
        is likely to read from recall (they stay in the episode log)."""
        now = _date(at)
        raw = [ep for ep in self._episodes if ep.state == "raw"]
        ids = {id(ep): ep.text if [e.text for e in raw].count(ep.text) == 1 else f"{ep.text} #{i}"
               for i, ep in enumerate(raw)}
        result = plan([self._unit(ep, ids[id(ep)], now) for ep in raw], self.budget, upgrades=("graph",))
        for ep in raw:
            action = result.actions[ids[id(ep)]]
            if action == "graph":
                ep.state = "graph"
                self._derive(ep.user, ep.text, ep.when)
            elif action == "forget":
                ep.state = "cold"
        return result

    def _unit(self, ep: Episode, uid: str, now: date) -> Unit:
        age = max((now - ep.when).days, 0)
        lowered = ep.text.lower()
        mentions = sum(1 for e in self.graph.entities() if e.id != ep.user and e.name.lower() in lowered)
        p = self.access(ep.user, ep.text, age) if self.access else prior_access(ep.text, age, mentions)
        if self.conflict:
            chi = self.conflict(ep.user, ep.text)
        else:  # a first-person turn about a one-valued relation the user already has may supersede it
            words = set(re.findall(r"[a-z]+", lowered))
            held = {e.relation for e in self.graph.edges(subject=ep.user) if e.valid_to is None} & FUNCTIONAL
            personal = bool(_FIRST_PERSON.search(ep.text))
            chi = 1.0 if personal and any(words & _RELATION_CUES.get(r, set()) for r in held) else 0.0
        return Unit(uid, int(len(ep.text.split()) * 1.3) + 1, p, mentions, chi, age)

    def _derive(self, user: str, text: str, when: date) -> list[Edge]:
        if self.graph.entity(user) is None:
            self.graph.upsert_entity(Entity(user, user, "person"))
        changed: list[Edge] = []
        for item in self.extractor.extract(user, text, when):
            changed += self.resolver.apply(item, user, when, source=text)
        if changed:
            self._stale = True
            if self.community_refresh == "on_change":
                self.refresh_communities(when)
        return changed

    def delete(self, user: str, text: str, at: str | date | None = None) -> int:
        """Delete the episodes with this text (on day ``at``, if given) and rebuild everything derived from
        them by replaying the remaining episodes in order. Returns how many episodes were deleted.

        Raises RuntimeError when the store holds facts from episodes this Memory has not seen (a store
        opened from another process), since those could not be rebuilt.
        """
        day = _date(at) if at is not None else None
        doomed = [ep for ep in self._episodes if ep.user == user and ep.text == text and day in (None, ep.when)]
        if not doomed:
            return 0
        known = {ep.text for ep in self._episodes}
        foreign = {e.source for e in self.graph.edges(include_retracted=True)} - known
        if foreign:
            raise RuntimeError(f"cannot rebuild: the store holds facts from {len(foreign)} unseen episodes")
        self._episodes = [ep for ep in self._episodes if all(ep is not d for d in doomed)]
        self.graph.clear()
        self.vectors.clear()
        for ep in self._episodes:
            if ep.state == "graph":
                self._derive(ep.user, ep.text, ep.when)
        self.epoch += 1
        self._stale = True
        return len(doomed)

    def refresh_communities(self, at: str | date | None = None) -> None:
        self._communities = comm.detect(self.graph, _date(at))
        self._communities_epoch = self.epoch
        self._stale = False

    def recall(self, user: str, query: str, at: str | date | None = None) -> Recall:
        now = _date(at)
        if (self._stale and self.community_refresh == "on_change") or self._communities_epoch != self.epoch:
            self.refresh_communities(now)  # communities built before a deletion are never served after it
        names = {e.id: e.name for e in self.graph.entities()}
        mine = [c for c in self._communities if user in c.members]
        words = _content(query)
        raw = [f"- {ep.text} (said {ep.when.isoformat()})" for ep in self._episodes
               if ep.user == user and ep.state == "raw" and words & _content(ep.text)]
        recalled = pack(self.graph.edges(subject=user), mine, query, now, names, self.recall_tokens, raw)
        recalled.epoch = self.epoch
        return recalled

    def export_notes(self, user: str | None = None) -> str:
        names = {e.id: e.name for e in self.graph.entities()}
        return notes.dump(self.graph.edges(subject=user, include_retracted=True), names)
