"""Deleting an episode removes everything built from it: edges, the edges it closed or corrected,
entities only it mentioned, and the community summaries over them. Derived artefacts carry an epoch,
so nothing built before a deletion is served after it."""

from datetime import date

import pytest

from graph_memory import Memory
from graph_memory.store import InMemoryGraphStore


def _history(m, relation):
    edges = m.graph.edges(subject="u42", relation=relation, include_retracted=True)
    return sorted((m.graph.entity(e.object).name, e.valid_from, e.valid_to) for e in edges)


def test_deleting_a_move_reopens_the_old_address():
    m = Memory()
    m.add(user="u42", text="I live in Manchester.", at="2025-06-03")
    m.add(user="u42", text="I moved to Leeds in March.", at="2026-04-02")
    m.add(user="u42", text="I work at Northwind Traders.", at="2026-04-05")
    assert m.delete(user="u42", text="I moved to Leeds in March.") == 1
    assert _history(m, "lives_in") == [("Manchester", date(2025, 6, 3), None)]
    assert _history(m, "works_at") == [("Northwind Traders", date(2026, 4, 5), None)]
    assert "Leeds" not in m.export_notes("u42")
    assert all(e.name != "Leeds" for e in m.graph.entities())


def test_recall_after_a_deletion_is_rebuilt_and_carries_the_new_epoch():
    m = Memory()
    m.add(user="u42", text="I live in Manchester.", at="2025-06-03")
    m.add(user="u42", text="I moved to Leeds in March.", at="2026-04-02")
    stale = m.recall(user="u42", query="where do I live", at="2026-05-01")
    assert "Leeds" in stale.context and stale.epoch == 0
    m.delete(user="u42", text="I moved to Leeds in March.")
    fresh = m.recall(user="u42", query="where do I live", at="2026-05-01")
    assert fresh.epoch == m.epoch == 1
    assert "Leeds" not in fresh.context
    assert all("Leeds" not in c.summary for c in fresh.communities)


def test_deleting_a_correction_restores_the_corrected_date():
    m = Memory()
    m.add(user="u42", text="I moved to Leeds in March.", at="2026-04-02")
    m.add(user="u42", text="Actually it was February, not March.", at="2026-04-09")
    m.delete(user="u42", text="Actually it was February, not March.")
    assert _history(m, "lives_in") == [("Leeds", date(2026, 3, 1), None)]


def test_an_unknown_episode_changes_nothing():
    m = Memory()
    m.add(user="u42", text="I live in York.", at="2026-01-05")
    assert m.delete(user="u42", text="I live in Paris.") == 0
    assert m.epoch == 0


def test_a_store_holding_facts_this_memory_never_saw_is_not_rebuilt():
    graph = InMemoryGraphStore()
    Memory(graph=graph).add(user="u1", text="I work at Acme.", at="2026-01-05")
    reopened = Memory(graph=graph)
    reopened.add(user="u1", text="I live in York.", at="2026-02-05")
    with pytest.raises(RuntimeError, match="cannot rebuild"):
        reopened.delete(user="u1", text="I live in York.")
    assert len(graph.edges(subject="u1")) == 2
