"""The ingest -> memory -> retrieve -> answer -> verify path, end to end and offline."""

from pathlib import Path

import pytest

from hana_knowledge import KnowledgeBase
from retrieval_kit.agent import Claim, Draft

FILINGS = Path(__file__).resolve().parents[2] / "examples" / "filings"


@pytest.fixture
def kb():
    kb = KnowledgeBase(FILINGS)
    kb.remember("u42", "I work at Northwind Traders.", at="2026-01-10")
    kb.remember("u42", "I joined Harbour Logistics in March.", at="2026-04-02")
    return kb


def test_memory_picks_the_employer_valid_at_the_time(kb):
    before = kb.ask("What was the operating margin of my employer?", user="u42", at="2026-02-01")
    after = kb.ask("What was the operating margin of my employer?", user="u42", at="2026-05-01")
    assert "18.4%" in before.text and before.citations[0].doc == "northwind-2024.txt"
    assert "7.2%" in after.text and after.citations[0].doc == "harbour-h1-2026.txt"
    assert before.verified and after.verified
    assert "Harbour Logistics" in after.memory_context


def test_citation_span_points_at_the_page_text(kb):
    answer = kb.ask("What does clause 14.2 limit?")
    c = answer.citations[0]
    page = (FILINGS / c.doc).read_text().split("\f")[c.page - 1]
    assert page[c.span[0]:c.span[1]] == c.text


def test_abstains_rather_than_guessing(kb):
    answer = kb.ask("Who won the cup final in 1966?")
    assert answer.abstained and answer.citations == []


class WrongArithmetic:
    """A generator that cites the right passage but restates its figure wrongly."""

    def draft(self, question, evidence):
        for i, passage in enumerate(evidence):
            if "Net debt" in passage:
                return Draft(claims=[Claim("Net debt was £310bn at the end of June.", i, "Net debt was £310m")])
        return Draft(abstain=True)


def test_cite_check_catches_and_revises_a_wrong_figure():
    kb = KnowledgeBase(FILINGS, model=WrongArithmetic())
    answer = kb.ask("What was Harbour's net debt?")
    assert answer.revised
    assert "£310m" in answer.text and "£310bn" not in answer.text
    assert answer.verified
