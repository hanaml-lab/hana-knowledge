"""Entities and bitemporal edges.

Every edge carries two time axes:

* valid time, ``valid_from`` / ``valid_to``: when the fact held in the world;
* recorded time, ``recorded_at`` / ``retracted_at``: when the agent learned it, and when it
  learned that this version of the fact was wrong.

Nothing is deleted. A move closes the old address's ``valid_to``; a correction retracts the
wrong version and records a new one, so both "where did they live in January?" and "what did
we believe on 3 April?" have answers.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field, replace
from datetime import date

_ids = itertools.count(1)


@dataclass(frozen=True)
class Entity:
    id: str
    name: str
    kind: str = "thing"


@dataclass(frozen=True)
class Edge:
    subject: str  # entity id
    relation: str
    object: str  # entity id
    valid_from: date | None
    valid_to: date | None
    recorded_at: date
    retracted_at: date | None = None
    source: str = ""
    id: int = field(default_factory=lambda: next(_ids))

    def valid_at(self, when: date) -> bool:
        starts = self.valid_from is None or self.valid_from <= when
        ends = self.valid_to is None or when < self.valid_to
        return starts and ends

    def known_at(self, when: date) -> bool:
        return self.recorded_at <= when and (self.retracted_at is None or when < self.retracted_at)

    @property
    def current(self) -> bool:
        return self.retracted_at is None

    def evolve(self, **changes) -> Edge:
        return replace(self, id=next(_ids), **changes)
