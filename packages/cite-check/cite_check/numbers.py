"""Recompute figures in code instead of trusting a model's arithmetic.

A source may hold a table as CSV (``name.csv``) or as a Markdown table. Claims about totals,
growth rates and shares are recomputed from it. DuckDB runs ``queries/figures.sql`` when it is
installed; the same arithmetic in Python is the fallback, and the two are tested to agree.
"""

from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass
from pathlib import Path

from cite_check.units import Quantity, close, parse

Table = dict[str, dict[str, float]]  # row label -> column label -> value


def read_table(text: str) -> Table:
    """A CSV or Markdown table whose first column holds row labels."""
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    if lines and lines[0].startswith("|"):
        rows = [[c.strip() for c in ln.strip("|").split("|")] for ln in lines
                if not re.fullmatch(r"\|?[\s:|-]+\|?", ln)]
    else:
        rows = list(csv.reader(io.StringIO("\n".join(lines))))
    header, body = rows[0], rows[1:]
    table: Table = {}
    for row in body:
        values = {}
        for col, cell in zip(header[1:], row[1:], strict=False):
            q = parse(cell)
            if q:
                values[col] = q[0].value
        table[row[0].strip().lower()] = values
    return table


@dataclass(frozen=True)
class Recomputed:
    kind: str  # growth | total | value
    expected: float
    stated: Quantity
    ok: bool
    detail: str


def growth(table: Table, row: str, start: str, end: str) -> float:
    a, b = table[row][start], table[row][end]
    return (b - a) / a * 100.0


def _find_row(table: Table, claim: str) -> str | None:
    lowered = claim.lower()
    hits = [r for r in table if r and r in lowered]
    return max(hits, key=len) if hits else None


def check_claim(claim: str, table: Table, tolerance: float = 0.005) -> list[Recomputed]:
    """Recompute what a claim states from a table.

    Recognised forms: "X grew/rose/fell N% from C1 to C2" (growth), "total X was N" or "X
    totalled N" (sum of a row), and "X was N in C" (a cell).
    """
    figures = parse(claim)
    row = _find_row(table, claim)
    if row is None or not figures:
        return []
    cols = list(table[row])
    mentioned = [c for c in cols if c.lower() in claim.lower()]
    out: list[Recomputed] = []
    pct = [q for q in figures if q.unit == "percent"]
    if pct and re.search(r"\b(grew|rose|increased|fell|declined|decreased|up|down)\b", claim, re.I):
        if len(mentioned) >= 2:
            start, end = mentioned[0], mentioned[-1]
            if claim.lower().find(start.lower()) > claim.lower().find(end.lower()):
                start, end = end, start
        elif len(cols) >= 2:
            start, end = cols[-2], cols[-1]
        else:
            return []
        expected = growth(table, row, start, end)
        stated = pct[0].value * (-1 if re.search(r"\b(fell|declined|decreased|down)\b", claim, re.I) else 1)
        ok = abs(expected - stated) <= max(0.05, abs(expected) * tolerance)
        out.append(Recomputed("growth", expected, pct[0], ok,
                              f"{row}: {start} -> {end} is {expected:+.1f}%, stated {stated:+.1f}%"))
        return out
    values = [q for q in figures if q.unit != "percent"]
    if not values:
        return out
    if re.search(r"\btotal", claim, re.I):
        expected = sum(table[row].values())
        out.append(Recomputed("total", expected, values[0], close(expected, values[0].value, tolerance),
                              f"{row}: total is {expected:g}, stated {values[0].value:g}"))
    elif mentioned:
        expected = table[row][mentioned[0]]
        out.append(Recomputed("value", expected, values[0], close(expected, values[0].value, tolerance),
                              f"{row} in {mentioned[0]} is {expected:g}, stated {values[0].value:g}"))
    return out


def growth_sql(path: Path, row: str, start: str, end: str) -> float:
    """The same growth rate computed by DuckDB over a CSV, using ``queries/figures.sql``."""
    import duckdb

    sql = (Path(__file__).resolve().parent.parent / "queries" / "figures.sql").read_text()
    con = duckdb.connect()
    con.execute(sql.split("-- name: growth")[0])  # the macros
    con.execute("CREATE TABLE t AS SELECT * FROM read_csv_auto(?, header = true, all_varchar = true)", [str(path)])
    label = con.execute("SELECT column_name FROM (DESCRIBE t) LIMIT 1").fetchone()[0]
    value = con.execute(
        f'SELECT growth_pct(CAST("{start}" AS DOUBLE), CAST("{end}" AS DOUBLE)) FROM t WHERE lower("{label}") = ?',
        [row.lower()],
    ).fetchone()[0]
    return float(value)
