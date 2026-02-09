from datetime import date

from graph_memory.communities import detect, louvain
from graph_memory.model import Edge, Entity
from graph_memory.recall import as_of
from graph_memory.store import InMemoryGraphStore


def test_louvain_splits_two_cliques():
    a = {"a", "b", "c"}
    b = {"x", "y", "z"}
    adj = {n: {} for n in a | b}
    for group in (a, b):
        for u in group:
            for v in group:
                if u != v:
                    adj[u][v] = 1.0
    adj["c"]["x"] = adj["x"]["c"] = 1.0
    parts = louvain(adj)
    assert parts["a"] == parts["b"] == parts["c"]
    assert parts["x"] == parts["y"] == parts["z"]
    assert parts["a"] != parts["x"]


def test_detect_uses_only_valid_edges():
    g = InMemoryGraphStore()
    for ent in (Entity("u", "u"), Entity("p:york", "York"), Entity("p:leeds", "Leeds")):
        g.upsert_entity(ent)
    g.add_edge(Edge("u", "lives_in", "p:york", date(2020, 1, 1), date(2026, 1, 1), date(2020, 1, 1)))
    g.add_edge(Edge("u", "lives_in", "p:leeds", date(2026, 1, 1), None, date(2026, 1, 1)))
    summaries = " ".join(c.summary for c in detect(g, date(2026, 6, 1)))
    assert "Leeds" in summaries and "York" not in summaries


def test_as_of():
    now = date(2026, 5, 1)
    assert as_of("where do I live", now) == now
    assert as_of("where did I live last year", now) == date(2025, 5, 1)
    assert as_of("where was I in March 2025", now) == date(2025, 3, 15)
    assert as_of("my job in 2024", now) == date(2024, 7, 1)
