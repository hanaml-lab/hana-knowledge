"""Chunks keep the path of the section they sit in, so the answer loop can open a section or follow a
cross-reference by its number instead of hoping a search finds it."""

import pytest

from retrieval_kit import Index

AGREEMENT = (
    "CREDIT AGREEMENT\n"
    "ARTICLE 1 DEFINITIONS\n"
    '1.1 Defined terms. "Consolidated EBITDA" means net income plus interest, taxes, depreciation and '
    "amortisation, excluding one-off restructuring charges.\n"
    "ARTICLE 7 COVENANTS\n"
    "7.11 Financial covenants. The Leverage Ratio shall not exceed 3.50 to 1.00 at the end of any quarter, "
    "with Consolidated EBITDA as defined in Section 1.1.\n"
    "7.12 Reporting. The Borrower delivers quarterly accounts within 45 days.\f"
    "7.13 Amendments. The threshold in Section 7.11 was raised to 4.00 to 1.00 by the First Amendment.\n"
)


@pytest.fixture
def agreement(tmp_path):
    (tmp_path / "credit-agreement.txt").write_text(AGREEMENT, encoding="utf-8")
    return Index.build(tmp_path, chunk_tokens=200, overlap=10)


def _with(index, text):
    return next(c for c in index.chunks if text in c.text)


def test_chunks_carry_their_section_path(agreement):
    assert _with(agreement, "Leverage Ratio").section == "ARTICLE 7 COVENANTS / 7.11 Financial covenants"
    definitions = "ARTICLE 1 DEFINITIONS / 1.1 Defined terms"
    assert _with(agreement, "Consolidated EBITDA\" means").section == definitions


def test_a_chunk_never_crosses_a_heading(agreement):
    leverage = _with(agreement, "Leverage Ratio")
    assert "Reporting" not in leverage.text and "Defined terms" not in leverage.text


def test_section_path_continues_across_pages(agreement):
    amendment = _with(agreement, "First Amendment")
    assert amendment.page == 2
    assert amendment.section == "ARTICLE 7 COVENANTS / 7.13 Amendments"


def test_open_a_section_by_its_number(agreement):
    opened = agreement.section("Section 1.1")
    assert [c.section for c in opened] == ["ARTICLE 1 DEFINITIONS / 1.1 Defined terms"]
    article = agreement.section("Article 7")
    assert len(article) == 3 and all(c.section.startswith("ARTICLE 7") for c in article)
    assert agreement.section("Section 9.9") == []


def test_saved_index_keeps_sections(agreement, tmp_path):
    agreement.save(tmp_path / "chunks.jsonl")
    loaded = Index.load(tmp_path / "chunks.jsonl")
    assert [c.section for c in loaded.chunks] == [c.section for c in agreement.chunks]


def test_plain_text_without_headings_is_one_unnamed_section(index):
    assert {c.section for c in index.chunks} == {""}
