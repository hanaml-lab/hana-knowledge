"""Review results and their renderings."""

from __future__ import annotations

import html
import json
from dataclasses import asdict, dataclass, field

from cite_check.claims import Claim


@dataclass
class Issue:
    claim: str
    kind: str  # unsupported | numeric | unit | contradicted
    evidence: str
    claim_id: int
    detail: str = ""


@dataclass
class Review:
    report: str
    claims: list[Claim]
    issues: list[Issue] = field(default_factory=list)
    evidence: dict[int, str] = field(default_factory=dict)  # claim id -> best passage

    @property
    def flagged(self) -> set[int]:
        return {i.claim_id for i in self.issues}

    def to_json(self) -> str:
        return json.dumps({
            "claims": len(self.claims),
            "issues": [asdict(i) for i in self.issues],
        }, indent=2)

    def to_markdown(self) -> str:
        lines = ["# cite-check review", "", f"{len(self.claims)} claims, {len(self.issues)} issues.", ""]
        for i in self.issues:
            lines += [f"- **{i.kind}**: {i.claim}", f"  - evidence: {i.evidence or '(none found)'}"]
            if i.detail:
                lines.append(f"  - {i.detail}")
        return "\n".join(lines) + "\n"

    def to_html(self) -> str:
        rows = "".join(
            f"<tr><td>{html.escape(i.kind)}</td><td>{html.escape(i.claim)}</td>"
            f"<td>{html.escape(i.evidence)}</td><td>{html.escape(i.detail)}</td></tr>" for i in self.issues)
        return ("<!doctype html><meta charset=utf-8><title>cite-check review</title>"
                "<style>body{font:15px system-ui;margin:2rem;max-width:70rem}td,th{border-bottom:1px solid #ddd;"
                "padding:.4rem;vertical-align:top;text-align:left}</style>"
                f"<h1>cite-check review</h1><p>{len(self.claims)} claims, {len(self.issues)} issues.</p>"
                f"<table><tr><th>Kind</th><th>Claim</th><th>Evidence</th><th>Detail</th></tr>{rows}</table>")
