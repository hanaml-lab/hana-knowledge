import importlib.util

import pytest

from cite_check.numbers import check_claim, growth, growth_sql, read_table

CSV = "segment,2023,2024\nCloud,1200,1500\nDevices,800,760\n"
MD = "| segment | 2023 | 2024 |\n|---|---|---|\n| Cloud | 1,200 | 1,500 |\n| Devices | 800 | 760 |\n"


def test_csv_and_markdown_tables_agree():
    assert read_table(CSV) == read_table(MD)


def test_growth_recomputed():
    t = read_table(CSV)
    assert growth(t, "cloud", "2023", "2024") == pytest.approx(25.0)
    ok = check_claim("Cloud revenue grew 25% from 2023 to 2024.", t)
    assert ok[0].kind == "growth" and ok[0].ok
    bad = check_claim("Cloud revenue grew 30% from 2023 to 2024.", t)
    assert not bad[0].ok


def test_decline_sign():
    t = read_table(CSV)
    assert check_claim("Devices revenue fell 5% in 2024.", t)[0].ok
    assert not check_claim("Devices revenue rose 5% in 2024.", t)[0].ok


def test_total_and_cell():
    t = read_table(CSV)
    assert check_claim("Total Cloud revenue was 2,700 over the two years.", t)[0].ok
    assert not check_claim("Devices revenue was 900 in 2024.", t)[0].ok


@pytest.mark.skipif(importlib.util.find_spec("duckdb") is None, reason="duckdb not installed")
def test_sql_matches_python(tmp_path):
    path = tmp_path / "segments.csv"
    path.write_text(CSV)
    assert growth_sql(path, "Cloud", "2023", "2024") == pytest.approx(growth(read_table(CSV), "cloud", "2023", "2024"))
