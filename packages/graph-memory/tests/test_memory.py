from datetime import date

from graph_memory import Memory
from graph_memory.notes import load


def test_move_closes_old_address():
    m = Memory()
    m.add(user="u42", text="I live in Manchester.", at="2025-06-03")
    m.add(user="u42", text="I moved to Leeds in March. Still at the same firm.", at="2026-04-02")
    open_edges = [e for e in m.graph.edges(subject="u42", relation="lives_in") if e.valid_to is None]
    assert [m.graph.entity(e.object).name for e in open_edges] == ["Leeds"]
    old = [e for e in m.graph.edges(subject="u42", relation="lives_in") if e.valid_to]
    assert old[0].valid_to == date(2026, 3, 1)


def test_correction_moves_both_ends():
    m = Memory()
    m.add(user="u42", text="I live in Manchester.", at="2025-06-03")
    m.add(user="u42", text="I moved to Leeds in March.", at="2026-04-02")
    m.add(user="u42", text="Actually it was February, not March.", at="2026-04-09")
    edges = {m.graph.entity(e.object).name: e for e in m.graph.edges(subject="u42")}
    assert edges["Leeds"].valid_from == date(2026, 2, 1)
    assert edges["Manchester"].valid_to == date(2026, 2, 1)
    # the wrong version is retracted, not deleted
    history = m.graph.edges(subject="u42", include_retracted=True)
    assert any(e.retracted_at == date(2026, 4, 9) and e.valid_from == date(2026, 3, 1) for e in history)


def test_recall_respects_valid_time():
    m = Memory()
    m.add(user="u42", text="I live in Manchester.", at="2025-01-03")
    m.add(user="u42", text="I moved to Leeds in February.", at="2026-04-02")
    now = m.recall(user="u42", query="Where does the user live, and since when?", at="2026-05-01")
    assert "Leeds (from 2026-02-01 to now" in now.context
    assert [m.graph.entity(e.object).name for e in now.edges] == ["Leeds"]
    before = m.recall(user="u42", query="Where did I live last year?", at="2026-05-01")
    assert [m.graph.entity(e.object).name for e in before.edges] == ["Manchester"]


def test_same_fact_twice_is_one_edge():
    m = Memory()
    m.add(user="u1", text="I work at Northwind.", at="2026-01-05")
    m.add(user="u1", text="I work at Northwind.", at="2026-02-05")
    assert len(m.graph.edges(subject="u1")) == 1


def test_resolver_merges_spelling_variants():
    m = Memory(resolve_threshold=0.75)
    m.add(user="u1", text="I work at Northwind Traders.", at="2026-01-05")
    m.add(user="u1", text="I work at Northwind Trader.", at="2026-02-05")
    assert len(m.graph.edges(subject="u1")) == 1


def test_recall_budget():
    m = Memory(recall_tokens=12)
    m.add(user="u1", text="I live in York. I work at Acme.", at="2026-01-05")
    assert len(m.recall(user="u1", query="where do I live", at="2026-02-01").context.split()) <= 12


def test_notes_round_trip():
    m = Memory()
    m.add(user="u42", text="I moved to Leeds in March.", at="2026-04-02")
    m.add(user="u42", text="Actually it was February, not March.", at="2026-04-09")
    rows = load(m.export_notes("u42"))
    assert {(r["object"], r["valid_from"], r["retracted_at"]) for r in rows} == {
        ("Leeds", date(2026, 3, 1), date(2026, 4, 9)),
        ("Leeds", date(2026, 2, 1), None),
    }
