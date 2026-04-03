"""Split a report into atomic claims, each tied to the citations around it.

Citations are written ``[source]`` or ``[source, p. 3]`` after the sentence they support, where
``source`` is a file name (with or without extension) in the sources folder. A sentence with
no citation inherits the citations of the paragraph's previous cited sentence, as reports
usually cite once per run of related sentences.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_CITE = re.compile(r"\[([^\[\]]+?)(?:,\s*p\.?\s*(\d+))?\]")
_SENT = re.compile(r"(?:\[[^\]\n]*\]|[^.!?\n\[]|[.!?](?=[\w\d%]))+[.!?]?(?:\s*\[[^\]]+\])*")
_CONJ = re.compile(r";\s+|,\s+(?:and|while|whereas|but)\s+(?=[A-Z]?[a-z]+\s+\w+)")


@dataclass(frozen=True)
class Source:
    name: str
    page: int | None = None


@dataclass(frozen=True)
class Claim:
    id: int
    text: str  # without citation markers
    sentence: str  # the sentence it came from, as written
    sources: tuple[Source, ...]
    start: int  # char offset of the sentence in the report
    end: int


def _strip(text: str) -> str:
    return re.sub(r"\s+", " ", _CITE.sub("", text)).strip(" ,;")


def split_claims(report: str, atomic: bool = True) -> list[Claim]:
    claims: list[Claim] = []
    for para_match in re.finditer(r"(?:[^\n]|\n(?!\s*\n))+", report):
        para, offset = para_match.group(), para_match.start()
        if para.lstrip().startswith("#"):
            continue  # headings are not claims
        inherited: tuple[Source, ...] = ()
        for m in _SENT.finditer(para):
            sentence = m.group().strip()
            if not _strip(sentence):
                continue
            sources = tuple(Source(n.strip(), int(p) if p else None) for n, p in _CITE.findall(sentence))
            if sources:
                inherited = sources
            parts = _CONJ.split(_strip(sentence)) if atomic else [_strip(sentence)]
            for part in parts:
                part = part.strip().rstrip(".") + "."
                if len(part.split()) >= 2:
                    claims.append(Claim(len(claims), part, sentence, sources or inherited,
                                        offset + m.start(), offset + m.end()))
    return claims
