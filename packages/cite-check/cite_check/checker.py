"""The review pipeline: claims -> passages -> support cascade -> recomputed figures."""

from __future__ import annotations

from pathlib import Path

from cite_check import nli as nli_mod
from cite_check.claims import split_claims
from cite_check.numbers import check_claim, read_table
from cite_check.report import Issue, Review
from cite_check.retrieve import SourceLibrary
from cite_check.revise import revise as revise_review
from cite_check.units import mixed_currencies, parse


class Checker:
    def __init__(self, nli: str = "lexical", verifier: str | None = None, reviewer: str | None = None,
                 nli_accept: float = 0.9, numeric_tolerance: float = 0.005, revise_passes: int = 1,
                 reranker: str | None = None):
        first = nli_mod.load(nli, "nli") or nli_mod.LexicalNLI()
        self.cascade = nli_mod.Cascade(first, nli_mod.load(verifier, "verifier"),
                                       nli_mod.load(reviewer, "reviewer"), accept=nli_accept)
        self.tolerance = numeric_tolerance
        self.revise_passes = revise_passes
        self.reranker = reranker
        self._library: SourceLibrary | None = None

    def check(self, report: str | Path, sources: str | Path | SourceLibrary) -> Review:
        text = Path(report).read_text(encoding="utf-8") if _is_path(report) else str(report)
        library = sources if isinstance(sources, SourceLibrary) else SourceLibrary(sources, reranker=self.reranker)
        self._library = library
        review = Review(text, split_claims(text))
        for claim in review.claims:
            review.issues += self._check_claim(claim, library, review)
        return review

    def _check_claim(self, claim, library: SourceLibrary, review: Review) -> list[Issue]:
        if not claim.sources:
            return [Issue(claim.text, "unsupported", "", claim.id, "no citation")]
        figures = parse(claim.text)
        if mixed_currencies(figures):
            return [Issue(claim.text, "unit", "", claim.id, "mixes currencies in one claim")]
        best = None
        for src in claim.sources:
            table_path = library.table(src.name)
            if table_path is not None:
                for r in check_claim(claim.text, read_table(table_path.read_text()), self.tolerance):
                    review.evidence[claim.id] = r.detail
                    if not r.ok:
                        kind = "unit" if _scale_slip(r.expected, r.stated.value) else "numeric"
                        return [Issue(claim.text, kind, r.detail, claim.id, f"expected {r.expected:g}")]
                    return []
            for ev in library.passages(claim.text, src.name, src.page):
                j = self.cascade.judge(claim.text, ev.text)
                if best is None or (j.entail, -j.contradict) > (best[0].entail, -best[0].contradict):
                    best = (j, ev)
        if best is None:
            return [Issue(claim.text, "unsupported", "", claim.id, "cited source not found")]
        judgement, ev = best
        review.evidence[claim.id] = ev.text
        if judgement.label == "supported":
            return []
        if judgement.label == "contradicted":
            kind = "numeric" if figures and parse(ev.text) else "contradicted"
            if kind == "numeric" and any(_scale_slip(f.value, c.value) for f in parse(ev.text) for c in figures):
                kind = "unit"
            return [Issue(claim.text, kind, ev.text, claim.id)]
        return [Issue(claim.text, "unsupported", ev.text, claim.id)]

    def revise(self, review: Review) -> Review:
        """Rewrite the flagged claims from their evidence, then check the result once more."""
        if self._library is None:
            raise RuntimeError("call check() first")
        current = review
        for _ in range(self.revise_passes):
            if not current.issues:
                break
            current = self.check(revise_review(current), self._library)
        return current


def _scale_slip(a: float, b: float) -> bool:
    """True when two figures differ by a power of 1,000: millions read as billions."""
    if a == 0 or b == 0:
        return False
    ratio = max(abs(a), abs(b)) / min(abs(a), abs(b))
    return any(abs(ratio - 10**k) / 10**k < 0.01 for k in (3, 6, 9))


def _is_path(value) -> bool:
    if isinstance(value, Path):
        return True
    return "\n" not in value and len(value) < 4096 and Path(value).is_file()
