"""The dated-notes export: a plain, tab-separated view of what an agent remembers.

    # graph-memory notes v1
    valid_from  valid_to  subject  relation  object  recorded_at  retracted_at

Empty cells mean "open" (no end) or "unknown". ``packages/notes-reader`` reads the same format
from TypeScript.
"""

from __future__ import annotations

from datetime import date

from graph_memory.model import Edge

HEADER = "# graph-memory notes v1"
COLUMNS = ["valid_from", "valid_to", "subject", "relation", "object", "recorded_at", "retracted_at"]


def _cell(value: date | str | None) -> str:
    if value is None:
        return ""
    text = value.isoformat() if isinstance(value, date) else value
    if "\t" in text or "\n" in text:
        raise ValueError(f"notes cells cannot contain tabs or newlines: {text!r}")
    return text


def dump(edges: list[Edge], names: dict[str, str]) -> str:
    rows = [HEADER, "\t".join(COLUMNS)]
    for e in sorted(edges, key=lambda e: (e.subject, e.valid_from or date.min, e.recorded_at)):
        rows.append("\t".join(_cell(v) for v in (
            e.valid_from, e.valid_to, names.get(e.subject, e.subject), e.relation,
            names.get(e.object, e.object), e.recorded_at, e.retracted_at)))
    return "\n".join(rows) + "\n"


def load(text: str) -> list[dict]:
    lines = [ln for ln in text.splitlines() if ln]
    if not lines or lines[0] != HEADER:
        raise ValueError("not a graph-memory notes v1 file")
    if lines[1].split("\t") != COLUMNS:
        raise ValueError("unexpected columns")
    out = []
    for ln in lines[2:]:
        cells = ln.split("\t")
        row = dict(zip(COLUMNS, cells, strict=True))
        for key in ("valid_from", "valid_to", "recorded_at", "retracted_at"):
            row[key] = date.fromisoformat(row[key]) if row[key] else None
        out.append(row)
    return out
