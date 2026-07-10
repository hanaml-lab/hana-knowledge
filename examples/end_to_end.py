"""The whole hana-knowledge pipeline, offline, on the three filings in examples/filings/.

    python examples/end_to_end.py

1. ingest the filings into a hybrid index;
2. tell graph-memory where the user works, then correct it;
3. ask questions that depend on that memory;
4. every answer is cited to a page span and re-checked by cite-check.
"""

from pathlib import Path

from hana_knowledge import KnowledgeBase

HERE = Path(__file__).parent


def main() -> None:
    kb = KnowledgeBase(HERE / "filings")
    kb.remember("u42", "I work at Northwind Traders.", at="2026-01-10")
    kb.remember("u42", "I joined Harbour Logistics in March.", at="2026-04-02")

    questions = [
        ("What was the operating margin of my employer in the half year?", "2026-05-01"),
        ("What was the operating margin of my employer in 2024?", "2026-02-01"),
        ("What does clause 14.2 limit?", None),
        ("Who won the cup final in 1966?", None),
    ]
    for question, when in questions:
        answer = kb.ask(question, user="u42", at=when)
        print(f"Q ({when or 'today'}): {question}")
        if answer.abstained:
            print("  abstained: no passage supports an answer")
        else:
            print(f"  A: {answer.text}")
            for c in answer.citations:
                print(f"     cites {c.doc} p.{c.page} chars {c.span[0]}-{c.span[1]}")
            print(f"  verified by cite-check: {answer.verified}")
        print()


if __name__ == "__main__":
    main()
