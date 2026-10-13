"""Document loading, chunking and the two indexes."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from retrieval_kit.index.bm25 import BM25Index, tokenize
from retrieval_kit.index.dense import DenseIndex, Encoder, HashingEncoder, load_encoder
from retrieval_kit.index.sections import Outline, matches, reference

__all__ = [
    "BM25Index", "Chunk", "DenseIndex", "Encoder", "HashingEncoder", "Index", "load_encoder", "tokenize",
]

_TOKEN = re.compile(r"\S+")


@dataclass(frozen=True)
class Chunk:
    """A passage of a document, with its position in the page text and the path of its section."""

    id: int
    doc: str
    page: int
    start: int  # char offset into the page text
    end: int
    text: str
    section: str = ""  # "ARTICLE 7 COVENANTS / 7.11 Financial covenants"; empty without headings


def _read_pages(path: Path) -> list[str]:
    """Return the text of each page. Form feeds separate pages in plain text files."""
    if path.suffix.lower() == ".pdf":
        try:
            from docling.document_converter import DocumentConverter
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("PDF parsing needs Docling: pip install 'retrieval-kit[pdf]'") from exc
        doc = DocumentConverter().convert(str(path)).document
        pages: dict[int, list[str]] = {}
        for item, _level in doc.iterate_items():
            text = getattr(item, "text", "")
            for prov in getattr(item, "prov", []) or []:
                pages.setdefault(prov.page_no, []).append(text)
        return ["\n".join(pages[k]) for k in sorted(pages)]
    return path.read_text(encoding="utf-8").split("\f")


def chunk_page(text: str, chunk_tokens: int, overlap: int) -> list[tuple[int, int]]:
    """Split a page into windows of whitespace tokens. Returns (start, end) char spans."""
    if overlap >= chunk_tokens:
        raise ValueError("overlap must be smaller than chunk_tokens")
    tokens = [m.span() for m in _TOKEN.finditer(text)]
    spans: list[tuple[int, int]] = []
    step = chunk_tokens - overlap
    for i in range(0, len(tokens), step):
        window = tokens[i : i + chunk_tokens]
        spans.append((window[0][0], window[-1][1]))
        if i + chunk_tokens >= len(tokens):
            break
    return spans


def _bm25(docs: list[str]):
    """The Tantivy extension when it is installed, else the pure-Python index."""
    try:
        from retrieval_kit_tantivy import Bm25Index as TantivyIndex
    except ImportError:
        return BM25Index(docs)
    return TantivyIndex(docs)


class Index:
    """BM25 and dense indexes over the same chunks."""

    def __init__(self, chunks: list[Chunk], encoder: Encoder | None = None):
        self.chunks = chunks
        self.bm25 = _bm25([c.text for c in chunks])
        self.encoder = encoder or HashingEncoder()
        self.dense = DenseIndex(self.encoder, [c.text for c in chunks])

    @classmethod
    def build(
        cls,
        source: str | Path,
        chunk_tokens: int = 384,
        overlap: int = 64,
        encoder: Encoder | str | None = None,
    ) -> Index:
        root = Path(source)
        files = sorted(p for p in ([root] if root.is_file() else root.rglob("*")) if p.is_file()
                       and p.suffix.lower() in {".txt", ".md", ".pdf"})
        chunks: list[Chunk] = []
        for path in files:
            doc = path.name if root.is_file() else str(path.relative_to(root))
            outline = Outline()  # chunks follow the document's sections and never cross a heading
            for page_no, text in enumerate(_read_pages(path), start=1):
                for lo, hi, section in outline.sections(text):
                    for start, end in chunk_page(text[lo:hi], chunk_tokens, overlap):
                        chunks.append(Chunk(len(chunks), doc, page_no, lo + start, lo + end,
                                            text[lo + start : lo + end], section))
        enc = load_encoder(encoder) if isinstance(encoder, str) else encoder
        return cls(chunks, enc)

    def section(self, ref: str, doc: str | None = None) -> list[Chunk]:
        """The chunks of the section a reference names: "Section 7.11", "Article VII", "clause 14.2", "7.11".

        A section includes its subsections. Empty when no heading carries that number.
        """
        found = reference(ref)
        if found is None:
            return []
        kind, label = found
        return [c for c in self.chunks if (doc is None or c.doc == doc) and matches(c.section, kind, label)]

    def save(self, path: str | Path) -> None:
        Path(path).write_text("\n".join(json.dumps(asdict(c)) for c in self.chunks), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path, encoder: Encoder | None = None) -> Index:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
        return cls([Chunk(**json.loads(line)) for line in lines if line], encoder)
