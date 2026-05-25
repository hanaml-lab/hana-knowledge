import json
from pathlib import Path

import pytest

from cite_check import Checker, split_claims
from cite_check.cli import main

ROWS = [json.loads(line) for line in Path(__file__).parent.parent.joinpath("data/injected-errors.jsonl").open()]


@pytest.fixture
def sources(tmp_path):
    for row in ROWS:
        (tmp_path / f"{row['source']}.txt").write_text(row["source_text"])
    (tmp_path / "segments.csv").write_text("segment,2023,2024\nCloud,1200,1500\n")
    return tmp_path


def test_split_claims_keeps_citations():
    claims = split_claims("Revenue rose [a, p. 2]. Margins held; costs fell. [b]\n\n# Heading\n\nNo cite here.")
    assert claims[0].sources[0].name == "a" and claims[0].sources[0].page == 2
    assert [c.text for c in claims[1:3]] == ["Margins held.", "costs fell."]
    assert claims[-1].sources == ()


@pytest.mark.parametrize("row", ROWS, ids=[r["id"] for r in ROWS])
def test_injected_errors(row, sources):
    review = Checker().check(row["report"], sources)
    kinds = [i.kind for i in review.issues]
    if row["injected"] is None:
        assert kinds == []
    else:
        assert row["injected"] in kinds


def test_table_recompute_and_revise(sources):
    checker = Checker()
    review = checker.check("Cloud revenue grew 30% from 2023 to 2024 [segments].", sources)
    assert review.issues[0].kind == "numeric"
    review = checker.check("Revenue was $4.2m in 2024 [northwind]. Net debt was £310m at the end of June [harbour].",
                           sources)
    assert [i.kind for i in review.issues] == ["unit"]
    revised = checker.revise(review)
    assert revised.issues == []
    assert "Net debt was £310m" in revised.report


def test_cli_exit_code(sources, tmp_path, capsys):
    report = tmp_path / "r.md"
    report.write_text("Net debt was £310bn at the end of June [harbour].")
    assert main([str(report), "--sources", str(sources), "--format", "json"]) == 1
    assert json.loads(capsys.readouterr().out)["issues"][0]["kind"] == "unit"
