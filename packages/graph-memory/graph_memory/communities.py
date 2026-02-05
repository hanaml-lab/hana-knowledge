"""Group related entities and keep a short summary per group.

Uses Leiden (``leidenalg``) when it is installed, otherwise the local-moving phase of Louvain,
which finds the same groups on the small, sparse graphs a single user's memory produces.
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from datetime import date

from graph_memory.model import Edge
from graph_memory.store import GraphStore


@dataclass(frozen=True)
class Community:
    id: int
    members: frozenset[str]
    summary: str


def louvain(adjacency: dict[str, dict[str, float]], seed: int = 0, max_passes: int = 20) -> dict[str, int]:
    """Assign each node to a community by greedily moving it to the neighbour community with
    the largest modularity gain, until no move improves modularity."""
    nodes = sorted(adjacency)
    community = {n: i for i, n in enumerate(nodes)}
    degree = {n: sum(adjacency[n].values()) for n in nodes}
    m2 = sum(degree.values())
    if m2 == 0:
        return community
    total = defaultdict(float)
    for n in nodes:
        total[community[n]] += degree[n]
    rng = random.Random(seed)
    for _ in range(max_passes):
        moved = False
        order = nodes[:]
        rng.shuffle(order)
        for n in order:
            current = community[n]
            total[current] -= degree[n]
            links = defaultdict(float)
            for nb, w in adjacency[n].items():
                if nb != n:
                    links[community[nb]] += w
            best, best_gain = current, links.get(current, 0.0) - total[current] * degree[n] / m2
            for c, w in links.items():
                gain = w - total[c] * degree[n] / m2
                if gain > best_gain + 1e-12:
                    best, best_gain = c, gain
            community[n] = best
            total[best] += degree[n]
            moved |= best != current
        if not moved:
            break
    relabel = {c: i for i, c in enumerate(sorted(set(community.values())))}
    return {n: relabel[c] for n, c in community.items()}


def _leiden(adjacency: dict[str, dict[str, float]]) -> dict[str, int] | None:
    try:
        import igraph as ig
        import leidenalg
    except ImportError:
        return None
    nodes = sorted(adjacency)
    idx = {n: i for i, n in enumerate(nodes)}
    pairs = {tuple(sorted((a, b))): w for a in adjacency for b, w in adjacency[a].items()}
    g = ig.Graph(n=len(nodes), edges=[(idx[a], idx[b]) for a, b in pairs])
    g.es["weight"] = list(pairs.values())
    part = leidenalg.find_partition(g, leidenalg.ModularityVertexPartition, weights="weight", seed=0)
    return {nodes[i]: c for c, members in enumerate(part) for i in members}


def summarise(edges: list[Edge], names: dict[str, str]) -> str:
    parts = []
    for e in sorted(edges, key=lambda e: (e.subject, e.relation)):
        since = f" since {e.valid_from.isoformat()}" if e.valid_from else ""
        parts.append(f"{names.get(e.subject, e.subject)} {e.relation.replace('_', ' ')} "
                     f"{names.get(e.object, e.object)}{since}")
    return "; ".join(parts)


def detect(graph: GraphStore, at: date) -> list[Community]:
    """Communities over the facts that hold at ``at``."""
    live = [e for e in graph.edges() if e.valid_at(at)]
    adjacency: dict[str, dict[str, float]] = defaultdict(dict)
    for e in live:
        adjacency[e.subject][e.object] = adjacency[e.subject].get(e.object, 0) + 1.0
        adjacency[e.object][e.subject] = adjacency[e.object].get(e.subject, 0) + 1.0
    assignment = _leiden(adjacency) or louvain(adjacency)
    names = {ent.id: ent.name for ent in graph.entities()}
    groups: dict[int, set[str]] = defaultdict(set)
    for node, c in assignment.items():
        groups[c].add(node)
    out = []
    for c, members in sorted(groups.items()):
        inside = [e for e in live if e.subject in members and e.object in members]
        out.append(Community(c, frozenset(members), summarise(inside, names)))
    return out
