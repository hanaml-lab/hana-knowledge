"""Chunks keep the path of the section they sit in, so the answer loop can open a section or follow a
cross-reference by its number instead of hoping a search finds it."""

import pytest

from retrieval_kit import Answerer, Index, Retriever
from retrieval_kit.agent import Claim, Draft

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


class ScriptedModel:
    """Plays a fixed list of drafts, recording the evidence it was shown each time."""

    def __init__(self, *drafts):
        self.drafts, self.seen = list(drafts), []

    def draft(self, question, evidence):
        self.seen.append(list(evidence))
        step = self.drafts.pop(0) if self.drafts else Draft(abstain=True)
        return step.decide(evidence) if isinstance(step, _Lazy) else step


def _quote_from(text):
    return lambda evidence: next(i for i, p in enumerate(evidence) if text in p)


def test_answer_loop_follows_a_cross_reference(agreement):
    def answer(evidence):
        i = _quote_from("means net income")(evidence)
        quote = "means net income plus interest, taxes, depreciation and amortisation"
        return Draft(claims=[Claim("Consolidated EBITDA is net income plus interest, taxes, depreciation and "
                                   "amortisation.", i, quote)])

    model = ScriptedModel(Draft(follow_reference="Section 1.1"), _Lazy(answer))
    result = Answerer(Retriever(agreement, k=1), model=model, max_rounds=3, abstain_threshold=0.0).ask(
        "What is the Leverage Ratio covenant?")
    assert not result.abstained
    assert result.citations[0].text.startswith("means net income")
    assert result.actions == ["search", "follow_reference", "answer"]


def test_answer_loop_opens_a_section(agreement):
    model = ScriptedModel(Draft(open_section="Article 7"), _Lazy(lambda ev: Draft(claims=[
        Claim("Quarterly accounts are due within 45 days.", _quote_from("45 days")(ev), "within 45 days")])))
    result = Answerer(Retriever(agreement, k=1), model=model, max_rounds=3, abstain_threshold=0.0).ask(
        "What does the Borrower report?")
    assert sum("45 days" in p for p in model.seen[1]) == 1  # the whole article is now in view
    assert result.citations[0].text == "within 45 days"


def test_an_unknown_section_falls_back_to_search(agreement):
    model = ScriptedModel(Draft(follow_reference="clause 14.2"), Draft(abstain=True))
    answerer = Answerer(Retriever(agreement, k=1), model=model, max_rounds=3, abstain_threshold=0.0)
    result = answerer.ask("Liability?")
    assert result.queries[-1] == "clause 14.2"
    assert result.abstained


def test_claims_whose_quote_is_not_in_the_evidence_are_reported(agreement):
    model = ScriptedModel(Draft(claims=[
        Claim("The ratio may not exceed 3.50 to 1.00.", 0, "shall not exceed 3.50 to 1.00"),
        Claim("The ratio was waived in 2025.", 0, "the covenant was waived"),
    ]), Draft(abstain=True))
    result = Answerer(Retriever(agreement, k=1), model=model, max_rounds=3, abstain_threshold=0.0).ask(
        "What is the Leverage Ratio limit?")
    assert result.text == "The ratio may not exceed 3.50 to 1.00."
    assert result.unanswered == ["The ratio was waived in 2025."]


class _Lazy:
    """A draft decided from the evidence the model is shown."""

    def __init__(self, decide):
        self.decide = decide
