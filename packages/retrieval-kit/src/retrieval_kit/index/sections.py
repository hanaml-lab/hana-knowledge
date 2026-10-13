"""Section headings in plain text: Markdown headings, "ARTICLE 7 COVENANTS", "Section 7.11 Financial
covenants", "Note 14 Leases" and numbered clauses such as "7.11 Financial covenants. The Leverage Ratio ...".

A document is read as a stack of headings. Each heading closes the headings at its level or deeper, so a
chunk's path reads like ``ARTICLE 7 COVENANTS / 7.11 Financial covenants``, and the stack carries over
page breaks.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_MARKDOWN = re.compile(r"^(?P<hashes>#{1,6})\s+(?P<title>\S.*)$")
# A named heading has a title that starts with a capital, or none at all:
# "Clause 14.2 of the agreement" is a sentence, not a heading.
_NAMED = re.compile(
    r"^(?P<kind>(?i:article|section|clause|note|schedule|exhibit))\s+"
    r"(?P<label>[0-9]+(?:\.[0-9]+)*(?!\.?[0-9])|(?i:[ivxlc]+)\b)"
    r"(?:[\s.:\u2013-]+(?P<title>[A-Z][^.\n]{0,80}))?\s*(?=$|\.(?:\s|$))"
)
_NUMBERED = re.compile(r"^(?P<label>[0-9]+(?:\.[0-9]+)+(?!\.?[0-9]))\s+(?P<title>[A-Z][^.\n]{0,80})")
_REFERENCE = re.compile(
    r"\b(?P<kind>(?i:article|section|clause|note|schedule|exhibit))\s+(?P<label>[0-9]+(?:\.[0-9]+)*|[IVXLC]+)\b"
)
_ROMAN = {"i": 1, "v": 5, "x": 10, "l": 50, "c": 100}
_TOP_LEVEL = {"article", "schedule", "exhibit", "note"}


@dataclass(frozen=True)
class Heading:
    level: int
    kind: str | None  # article, section, clause, note, schedule, exhibit; None for Markdown and "7.11 ..."
    label: str | None  # "7.11", "7"; roman numerals are converted
    text: str  # the heading as it appears in a section path


def _label(raw: str) -> str:
    if re.fullmatch(r"[ivxlc]+", raw, re.IGNORECASE):
        values = [_ROMAN[ch] for ch in raw.lower()]
        return str(sum(-v if i + 1 < len(values) and v < values[i + 1] else v for i, v in enumerate(values)))
    return raw


def _match(line: str) -> tuple[Heading, int] | None:
    """The heading a line starts with and where it ends in the stripped line."""
    if m := _MARKDOWN.match(line):
        return Heading(len(m["hashes"]), None, None, m["title"].strip()), m.end()
    if m := _NAMED.match(line):
        kind, label = m["kind"].lower(), _label(m["label"])
        level = 1 if kind in _TOP_LEVEL else label.count(".") + 1
        return Heading(level, kind, label, m.group(0).strip().rstrip(" .:-")), m.end()
    if m := _NUMBERED.match(line):
        label = m["label"]
        return Heading(label.count(".") + 1, None, label, f"{label} {m['title'].strip()}"), m.end()
    return None


def parse_heading(line: str) -> Heading | None:
    """The heading a line starts with, or None. For a numbered clause only the label and title count."""
    found = _match(line.strip())
    return found[0] if found else None


def reference(text: str) -> tuple[str | None, str] | None:
    """(kind, label) of the first reference in ``text``.

    "Section 7.11" -> ("section", "7.11"); "Article VII" -> ("article", "7"); "7.11" -> (None, "7.11").
    """
    if m := _REFERENCE.search(text):
        return m["kind"].lower(), _label(m["label"])
    if m := re.search(r"\b[0-9]+(?:\.[0-9]+)*\b", text):
        return None, m.group(0)
    return None


def matches(path: str, kind: str | None, label: str) -> bool:
    """True when some heading in ``path`` is the referenced one. Articles and notes must match their kind;
    "Section 7.11" and "clause 7.11" both match a numbered "7.11 ..." line."""
    for part in path.split(" / ") if path else []:
        heading = parse_heading(part)
        if heading is None or heading.label != label:
            continue
        if kind in _TOP_LEVEL or heading.kind in _TOP_LEVEL:
            if heading.kind == kind:
                return True
        else:
            return True
    return False


class Outline:
    """Reads pages in order and splits each into (start, end, path) sections."""

    def __init__(self) -> None:
        self._stack: list[Heading] = []

    @property
    def path(self) -> str:
        return " / ".join(h.text for h in self._stack)

    def sections(self, page: str) -> list[tuple[int, int, str]]:
        out: list[tuple[int, int, str]] = []
        start = offset = 0
        body = False  # whether the open section has text beyond its heading lines
        for line in page.splitlines(keepends=True):
            found = _match(line.strip())
            if found is not None:
                heading, end = found
                if body:  # close the open section; a heading with no body yet joins the next one
                    out.append((start, offset, self.path))
                    start = offset
                while self._stack and self._stack[-1].level >= heading.level:
                    self._stack.pop()
                self._stack.append(heading)
                body = bool(line.strip()[end:].strip(" .:-"))
            elif line.strip():
                body = True
            offset += len(line)
        if offset > start:
            out.append((start, offset, self.path))
        return out
