"""Citations are character spans inside a page; they are checked before an answer leaves."""

from __future__ import annotations

from dataclasses import dataclass

from retrieval_kit.index import Chunk


@dataclass(frozen=True)
class Citation:
    doc: str
    page: int
    span: tuple[int, int]  # char offsets into the page text
    text: str


def cite(chunk: Chunk, start: int, end: int) -> Citation:
    """Cite chars [start, end) of a chunk, expressed as page offsets."""
    if not 0 <= start < end <= len(chunk.text):
        raise ValueError(f"span {start}:{end} is outside chunk {chunk.id}")
    return Citation(chunk.doc, chunk.page, (chunk.start + start, chunk.start + end), chunk.text[start:end])


def find_span(chunk: Chunk, quote: str) -> Citation | None:
    """Locate a quoted passage in a chunk, ignoring differences in whitespace."""
    norm_chunk, index_map = _squash(chunk.text)
    norm_quote, _ = _squash(quote)
    if not norm_quote:
        return None
    at = norm_chunk.find(norm_quote)
    if at < 0:
        return None
    start, end = index_map[at], index_map[at + len(norm_quote) - 1] + 1
    return cite(chunk, start, end)


def rechunk(citation: Citation, chunk: Chunk) -> tuple[int, int] | None:
    """Map a page-level citation into the local offsets of another chunk of the same page.

    Offsets are kept against the page, so a passage that is re-chunked with a different window
    still resolves to the same characters.
    """
    if (citation.doc, citation.page) != (chunk.doc, chunk.page):
        return None
    s, e = citation.span
    if s < chunk.start or e > chunk.end:
        return None
    return s - chunk.start, e - chunk.start


def verify(citation: Citation, pages: dict[tuple[str, int], str]) -> bool:
    """True when the cited span still holds exactly the cited text."""
    text = pages.get((citation.doc, citation.page))
    if text is None:
        return False
    s, e = citation.span
    return 0 <= s < e <= len(text) and text[s:e] == citation.text


def _squash(text: str) -> tuple[str, list[int]]:
    out, index_map, prev_space = [], [], False
    for i, ch in enumerate(text):
        if ch.isspace():
            if prev_space or not out:
                continue
            out.append(" ")
            index_map.append(i)
            prev_space = True
        else:
            out.append(ch)
            index_map.append(i)
            prev_space = False
    while out and out[-1] == " ":
        out.pop()
        index_map.pop()
    return "".join(out), index_map
