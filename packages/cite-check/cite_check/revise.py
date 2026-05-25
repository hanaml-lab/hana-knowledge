"""One verify-and-revise pass that rewrites only the flagged claims.

A numeric or unit error is fixed in place: the stated figure is replaced with the recomputed or
cited one, and nothing else in the sentence changes. An unsupported or contradicted claim is
replaced by the evidence sentence that covers it best, keeping the original citation, or
dropped when no evidence was found.
"""

from __future__ import annotations

import re

from cite_check.report import Review
from cite_check.units import parse

_SENT = re.compile(r"(?:[^.!?\n]|[.!?](?=[\w\d%]))+[.!?]?")


def _format_like(stated: str, value: float) -> str:
    q = parse(stated)[0]
    scale = q.value / float(re.sub(r"[^\d.\-]", "", stated.split()[0].lstrip("$£€")) or 1)
    number = value / scale if scale else value
    text = f"{number:,.1f}".rstrip("0").rstrip(".") if number % 1 else f"{number:,.0f}"
    return re.sub(r"-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|-?\d+(?:\.\d+)?", text, stated, count=1)


def _best_sentence(claim: str, evidence: str) -> str:
    words = set(re.findall(r"\w+", claim.lower()))
    sentences = [s.strip() for s in _SENT.findall(evidence) if s.strip()]
    return max(sentences, key=lambda s: len(words & set(re.findall(r"\w+", s.lower()))), default="")


def revise(review: Review) -> str:
    text = review.report
    by_claim = {c.id: c for c in review.claims}
    # Work from the end so earlier offsets stay valid.
    for issue in sorted(review.issues, key=lambda i: by_claim[i.claim_id].start, reverse=True):
        claim = by_claim[issue.claim_id]
        sentence = text[claim.start:claim.end]
        cites = "".join(re.findall(r"\s*\[[^\]]+\]", sentence))
        if issue.kind in {"numeric", "unit"} and issue.detail.startswith("expected "):
            expected = float(issue.detail.split()[1])
            stated = parse(claim.text)
            target = next((q for q in stated if q.unit != "percent"), stated[0]) if stated else None
            if target is not None:
                new = sentence.replace(target.text, _format_like(target.text, expected), 1)
                text = text[:claim.start] + new + text[claim.end:]
                continue
        if issue.kind in {"numeric", "unit", "contradicted", "unsupported"} and issue.evidence:
            replacement = _best_sentence(claim.text, issue.evidence).rstrip(".") + "." + cites
            text = text[:claim.start] + replacement + text[claim.end:]
        else:
            text = text[:claim.start] + text[claim.end:]
    return re.sub(r"[ \t]{2,}", " ", text)
