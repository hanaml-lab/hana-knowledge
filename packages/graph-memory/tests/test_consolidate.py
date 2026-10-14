"""Budgeted consolidation after TIDE (hanaML, 2026): which pending turns earn an expensive graph write.

The worked examples use the paper's default weights. A graph write costs 1.0 per 1k tokens and a summary
0.45; the value of an action is p * d_acc + lam_g * kappa * 0.01 * d_rel + lam_c * chi * d_conf - lam_s * store.
"""

import pytest

from graph_memory import Memory
from graph_memory.consolidate import Unit, p_within, plan


def _units():
    return [
        Unit("often-read", tokens=1000, p_access=0.9, kappa=0, chi=0.0),
        Unit("contradicts", tokens=1000, p_access=0.1, kappa=0, chi=0.9),
        Unit("small-talk", tokens=1000, p_access=0.2, kappa=0, chi=0.0),
    ]


def test_a_contradiction_is_consolidated_first_even_if_rarely_read():
    # graph gains: often-read 0.112, contradicts 0.2292, small-talk 0.035; only one graph write fits
    result = plan(_units(), budget=1 / 3)
    assert result.actions == {"often-read": "keep", "contradicts": "graph", "small-talk": "keep"}
    assert result.spent == pytest.approx(1.0) and result.eager == pytest.approx(3.0)


def test_a_larger_budget_buys_a_summary_for_the_often_read_turn():
    result = plan(_units(), budget=0.5)
    assert result.actions == {"often-read": "summary", "contradicts": "graph", "small-talk": "keep"}
    assert result.spent == pytest.approx(1.45)


def test_a_full_budget_writes_every_turn_with_a_positive_gain_to_the_graph():
    assert set(plan(_units(), budget=1.0).actions.values()) == {"graph"}


def test_old_unread_uncontested_turns_are_forgotten():
    stale = Unit("stale", tokens=200, p_access=0.01, kappa=0, chi=0.0, age=10)
    fresh = Unit("fresh", tokens=200, p_access=0.01, kappa=0, chi=0.0, age=2)
    assert plan([stale, fresh], budget=0.0).actions == {"stale": "forget", "fresh": "keep"}


def test_the_best_single_upgrade_beats_a_greedy_that_filled_up_on_cheap_items():
    cheap = Unit("cheap", tokens=100, p_access=0.9, kappa=0, chi=0.0)  # gain 0.112 for cost 0.1
    big = Unit("big", tokens=1000, p_access=0.5, kappa=0, chi=1.0)  # gain 0.296 for cost 1.0
    result = plan([cheap, big], budget=1.0 / 1.1, upgrades=("graph",))
    assert result.actions == {"cheap": "keep", "big": "graph"}


def test_access_within_a_horizon_from_per_step_hazards():
    assert p_within([0.5, 0.5]) == pytest.approx(0.75)
    assert p_within([]) == 0.0


def test_budgeted_memory_defers_the_graph_write_until_a_cycle():
    m = Memory(consolidation="budgeted", budget=1.0)
    m.add(user="u42", text="I live in Manchester.", at="2026-01-03")
    assert m.graph.edges(subject="u42") == []
    assert "Manchester" in m.recall(user="u42", query="where do I live", at="2026-01-04").context  # raw, still found
    m.consolidate(at="2026-01-04")
    assert [m.graph.entity(e.object).name for e in m.graph.edges(subject="u42")] == ["Manchester"]


def test_a_tight_budget_spends_the_graph_write_on_the_superseding_turn():
    m = Memory(consolidation="budgeted", budget=1.0)
    m.add(user="u42", text="I live in Manchester.", at="2026-01-03")
    m.consolidate(at="2026-01-03")
    m.budget = 0.6
    m.add(user="u42", text="I moved to Leeds in March.", at="2026-04-02")
    m.add(user="u42", text="The weather was lovely on the walk to the station this morning.", at="2026-04-02")
    actions = m.consolidate(at="2026-04-03").actions
    assert actions["I moved to Leeds in March."] == "graph"
    assert actions["The weather was lovely on the walk to the station this morning."] == "keep"
    lives = [e for e in m.graph.edges(subject="u42", relation="lives_in") if e.valid_to is None]
    assert [m.graph.entity(e.object).name for e in lives] == ["Leeds"]


def test_eager_memory_has_nothing_to_consolidate():
    m = Memory()
    m.add(user="u42", text="I live in Manchester.", at="2026-01-03")
    assert m.consolidate(at="2026-01-04").actions == {}
