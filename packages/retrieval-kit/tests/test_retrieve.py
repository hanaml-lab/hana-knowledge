from pathlib import Path

from retrieval_kit import Answerer, Config, Retriever
from retrieval_kit.agent import Claim, Draft
from retrieval_kit.index.bm25 import BM25Index


def test_bm25_prefers_rare_exact_terms():
    bm = BM25Index(["clause 14.2 limits liability", "the clause is long", "liability insurance"])
    assert bm.search("clause 14.2")[0][0] == 0


def test_retriever_finds_margin(index):
    hits = Retriever(index, k=3).search("operating margin fiscal 2023")
    assert "operating margin" in hits[0].chunk.text


def test_answer_cites_a_real_span(index):
    result = Answerer(Retriever(index), max_rounds=2).ask("What was the operating margin in fiscal 2023?")
    assert not result.abstained
    assert "18.4" in result.text
    c = result.citations[0]
    assert c.doc == "acme-10k.txt" and c.page == 1


def test_abstains_without_evidence(index):
    result = Answerer(Retriever(index)).ask("Who won the cup final in 1966?")
    assert result.abstained
    assert result.citations == []


class FollowUpThenAnswer:
    def __init__(self):
        self.calls = 0

    def draft(self, question, evidence):
        self.calls += 1
        if self.calls == 1:
            return Draft(follow_up="dividend per share")
        for i, passage in enumerate(evidence):
            if "dividend" in passage:
                return Draft(claims=[Claim("The dividend was 0.42 dollars.", i, "dividend of 0.42 dollars")])
        return Draft(abstain=True)


def test_follow_up_round_adds_evidence(index):
    result = Answerer(Retriever(index), model=FollowUpThenAnswer(), max_rounds=3,
                      abstain_threshold=0.3).ask("What was revenue in fiscal 2023?")
    assert result.queries == ["What was revenue in fiscal 2023?", "dividend per share"]
    assert result.rounds == 2
    assert result.citations[0].text == "dividend of 0.42 dollars"


def test_config_loads(tmp_path):
    cfg = Config.load(Path(__file__).parent.parent / "configs/financebench.yaml")
    assert cfg.rrf_k == 60 and cfg.max_rounds == 3
