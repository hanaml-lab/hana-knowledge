import pytest

from retrieval_kit.fuse import rrf


def test_item_in_both_lists_wins():
    fused = rrf([["a", "b", "c"], ["c", "d", "a"]], k=60)
    assert [item for item, _ in fused][:2] == ["a", "c"]
    assert fused[0][1] == pytest.approx(1 / 61 + 1 / 63)


def test_ties_break_on_best_rank_then_first_seen():
    fused = rrf([["a", "b"], ["b", "a"]], k=60)
    assert fused[0][1] == pytest.approx(fused[1][1])
    assert [item for item, _ in fused] == ["a", "b"]
    fused = rrf([["x"], ["y"]], k=0)
    assert [item for item, _ in fused] == ["x", "y"]


def test_empty_lists():
    assert rrf([]) == []
    assert rrf([[], []]) == []
    assert [i for i, _ in rrf([[], ["a"]])] == ["a"]


def test_duplicates_within_a_list_count_once():
    fused = dict(rrf([["a", "a", "b"]], k=60))
    assert fused["a"] == pytest.approx(1 / 61)


def test_negative_k_rejected():
    with pytest.raises(ValueError):
        rrf([["a"]], k=-1)
