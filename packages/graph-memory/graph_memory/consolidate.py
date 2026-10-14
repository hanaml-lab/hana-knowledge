"""Budgeted consolidation after TIDE (hanaML, 2026; reference implementation in
github.com/hanaml-lab/agent-memory-consolidation).

Writing a turn into the graph is the expensive step of a memory system: an extraction call per turn. Under a
budget, a turn earns that write only if it is likely to be read again, links to what is already known, or
contradicts a held belief. Each cycle solves a multiple-choice knapsack over the actions keep, summary, graph
and forget, greedily by marginal value per token, with the best-single-upgrade check that keeps the greedy
answer within half of the optimum.

    value(u, a) = p(u) * d_acc[a] + lam_g * kappa(u) * 0.01 * d_rel[a] + lam_c * chi(u) * d_conf[a]
                  - lam_s * store[a]

``p`` is the probability the turn is read within the horizon, ``kappa`` how many known entities it mentions and
``chi`` how strongly it contradicts a current belief. The weights are the paper's defaults. The LoRA tier of the
paper is not carried over.
"""

from __future__ import annotations

import heapq
import math
from collections.abc import Iterable
from dataclasses import dataclass, field

COST = {"keep": 0.0, "forget": 0.0, "summary": 0.45, "graph": 1.0}  # LLM tokens per token of the turn


@dataclass(frozen=True)
class Weights:
    lam_g: float = 0.15
    lam_c: float = 0.60
    lam_s: float = 0.02
    d_acc: dict = field(default_factory=lambda: {"keep": 0.0, "summary": 0.06, "graph": 0.11, "forget": -0.25})
    d_rel: dict = field(default_factory=lambda: {"keep": 0.0, "summary": 0.3, "graph": 1.0, "forget": 0.0})
    d_conf: dict = field(default_factory=lambda: {"keep": 0.0, "summary": 0.05, "graph": 0.38, "forget": 0.0})
    store: dict = field(default_factory=lambda: {"keep": 1.0, "summary": 0.2, "graph": 0.35, "forget": 0.0})
    forget_below: tuple[float, float, int] = (0.04, 0.10, 6)  # p, chi, minimum age


@dataclass(frozen=True)
class Unit:
    id: str
    tokens: int
    p_access: float
    kappa: float
    chi: float
    age: int = 0  # cycles (days, in Memory) since the turn arrived

    def cost(self, action: str) -> float:
        return COST[action] * self.tokens / 1000.0


@dataclass
class Plan:
    actions: dict[str, str]
    spent: float  # LLM tokens per 1k, as Unit.cost counts them
    eager: float  # what writing every turn to the graph would have cost


def value(unit: Unit, action: str, w: Weights) -> float:
    return (unit.p_access * w.d_acc[action] + w.lam_g * unit.kappa * 0.01 * w.d_rel[action]
            + w.lam_c * unit.chi * w.d_conf[action] - w.lam_s * w.store[action])


def p_within(hazards: Iterable[float]) -> float:
    """Probability of at least one access, from per-step hazards h_1..h_D: 1 - prod(1 - h_k)."""
    hs = list(hazards)
    return 1.0 - math.prod(1.0 - h for h in hs) if hs else 0.0


def plan(units: list[Unit], budget: float, weights: Weights | None = None,
         upgrades: tuple[str, ...] = ("summary", "graph")) -> Plan:
    """Choose an action per unit so that the cost stays within ``budget`` times eager consolidation."""
    w = weights or Weights()
    eager = sum(u.cost("graph") for u in units)
    cap = budget * eager + 1e-12
    p_min, chi_min, age_min = w.forget_below
    base = {u.id: "forget" if u.p_access < p_min and u.chi < chi_min and u.age >= age_min else "keep"
            for u in units}
    act = dict(base)
    heap: list[tuple[float, str, str, str]] = []

    def push(u: Unit) -> None:
        for a in upgrades:
            gain, extra = value(u, a, w) - value(u, act[u.id], w), u.cost(a) - u.cost(act[u.id])
            if gain > 0 and extra > 0:
                heapq.heappush(heap, (-gain / extra, u.id, a, act[u.id]))

    by_id = {u.id: u for u in units}
    for u in units:
        push(u)
    spent = 0.0
    while heap:
        _, uid, a, was = heapq.heappop(heap)
        u = by_id[uid]
        if act[uid] != was:
            continue  # stale: the unit was upgraded since this entry was pushed
        extra = u.cost(a) - u.cost(was)
        if spent + extra <= cap:
            spent, act[uid] = spent + extra, a
            push(u)

    # Best single upgrade: greedy by ratio can fill the budget with cheap items and miss one large one.
    greedy_gain = sum(value(by_id[i], act[i], w) - value(by_id[i], base[i], w) for i in act)
    single = max(((value(u, a, w) - value(u, base[u.id], w), u.id, a) for u in units for a in upgrades
                  if u.cost(a) <= cap), default=None)
    if single is not None and single[0] > greedy_gain:
        act = dict(base)
        act[single[1]] = single[2]
        spent = by_id[single[1]].cost(single[2])
    return Plan(act, spent, eager)
