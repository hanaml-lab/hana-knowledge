"""Turn a conversation turn into dated facts.

``PatternExtractor`` handles the common first-person statements with no model and is what the
tests use. ``LLMExtractor`` calls a small model (Llama 3.1 8B by default, served by SGLang or
vLLM) constrained to ``FACT_SCHEMA``.
"""

from __future__ import annotations

import json
import os
import re
import urllib.request
from dataclasses import dataclass
from datetime import date
from typing import Protocol

MONTHS = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august",
     "september", "october", "november", "december"], start=1)}


@dataclass(frozen=True)
class Fact:
    subject: str
    relation: str
    object: str
    object_kind: str
    valid_from: date | None = None
    keeps_previous: bool = False  # "still at the same firm": the open edge stays as it is


@dataclass(frozen=True)
class Correction:
    """The user corrected the start date of a recent fact ("it was February, not March")."""

    wrong: date
    right: date


class Extractor(Protocol):
    def extract(self, user: str, text: str, at: date) -> list[Fact | Correction]: ...


def month_date(name: str, at: date) -> date:
    """The most recent first-of-month called ``name`` that is not after ``at``."""
    month = MONTHS[name.lower()]
    year = at.year if month <= at.month else at.year - 1
    return date(year, month, 1)


_PLACE = r"([A-Z][\w'-]+(?: [A-Z][\w'-]+)*)"
_RULES: list[tuple[re.Pattern[str], str, str]] = [
    (re.compile(rf"\bI (?:moved|relocated) to {_PLACE}(?: in (\w+))?"), "lives_in", "place"),
    (re.compile(rf"\bI live in {_PLACE}"), "lives_in", "place"),
    (re.compile(rf"\bI (?:joined|started (?:at|with)|work (?:at|for|with)) {_PLACE}(?: in (\w+))?"), "works_at",
     "organisation"),
    (re.compile(rf"\bmy (?:partner|wife|husband) is (?:called )?{_PLACE}"), "partner", "person"),
]
_CORRECTION = re.compile(r"\b(?:actually )?it was (\w+),? not (\w+)", re.IGNORECASE)
_STILL = re.compile(r"\bstill (?:at|with) the same (firm|company|employer|job)", re.IGNORECASE)


class PatternExtractor:
    def extract(self, user: str, text: str, at: date) -> list[Fact | Correction]:
        out: list[Fact | Correction] = []
        for pattern, relation, kind in _RULES:
            for m in pattern.finditer(text):
                when = None
                if m.lastindex and m.lastindex >= 2 and m.group(2) and m.group(2).lower() in MONTHS:
                    when = month_date(m.group(2), at)
                out.append(Fact(user, relation, m.group(1), kind, when or at))
        for m in _CORRECTION.finditer(text):
            right, wrong = m.group(1).lower(), m.group(2).lower()
            if right in MONTHS and wrong in MONTHS:
                out.append(Correction(month_date(wrong, at), month_date(right, at)))
        if _STILL.search(text):
            out.append(Fact(user, "works_at", "", "organisation", keeps_previous=True))
        return out


FACT_SCHEMA = {
    "type": "object",
    "properties": {
        "facts": {"type": "array", "items": {
            "type": "object",
            "required": ["relation", "object", "object_kind"],
            "properties": {
                "relation": {"type": "string"},
                "object": {"type": "string"},
                "object_kind": {"type": "string"},
                "valid_from": {"type": ["string", "null"], "format": "date"},
            },
        }},
        "corrections": {"type": "array", "items": {
            "type": "object", "required": ["wrong", "right"],
            "properties": {"wrong": {"type": "string"}, "right": {"type": "string"}},
        }},
    },
    "required": ["facts"],
}


class LLMExtractor:
    def __init__(self, model: str, endpoint: str | None = None):
        self.model = model
        self.endpoint = (endpoint or os.environ.get("EXTRACTOR_URL", "http://localhost:30000/v1")).rstrip("/")

    def extract(self, user: str, text: str, at: date) -> list[Fact | Correction]:
        body = {
            "model": self.model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": (
                    "Extract facts the user states about themselves, with the date each became true. "
                    f"Today is {at.isoformat()}. Use snake_case relations such as lives_in, works_at.")},
                {"role": "user", "content": text},
            ],
            "response_format": {"type": "json_schema", "json_schema": {"name": "facts", "schema": FACT_SCHEMA}},
        }
        req = urllib.request.Request(f"{self.endpoint}/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(json.load(resp)["choices"][0]["message"]["content"])
        return parse(user, data, at)


def parse(user: str, data: dict, at: date) -> list[Fact | Correction]:
    out: list[Fact | Correction] = []
    for f in data.get("facts", []):
        vf = date.fromisoformat(f["valid_from"]) if f.get("valid_from") else at
        out.append(Fact(user, f["relation"], f["object"], f["object_kind"], vf))
    for c in data.get("corrections", []):
        out.append(Correction(date.fromisoformat(c["wrong"]), date.fromisoformat(c["right"])))
    return out
