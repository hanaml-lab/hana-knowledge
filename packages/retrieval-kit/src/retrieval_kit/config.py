"""Settings, loaded from YAML."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Config:
    chunk_tokens: int = 384
    overlap: int = 64
    fusion: str = "rrf"
    rrf_k: int = 60
    rerank_top: int = 50
    answer_top: int = 8
    max_rounds: int = 3
    abstain_threshold: float = 0.35
    dense: str | None = None
    reranker: str | None = None
    model: str | None = None

    @classmethod
    def load(cls, path: str | Path) -> Config:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        known = {f.name for f in fields(cls)}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"unknown settings: {', '.join(sorted(unknown))}")
        if raw.get("fusion", "rrf") != "rrf":
            raise ValueError("only fusion: rrf is supported")
        return cls(**raw)
