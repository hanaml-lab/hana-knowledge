"""Figures with their scale and unit: "$4.2bn", "4,200 million dollars", "18.4%", "12 kg"."""

from __future__ import annotations

import re
from dataclasses import dataclass

SCALES = {
    "k": 1e3, "thousand": 1e3,
    "m": 1e6, "mn": 1e6, "million": 1e6,
    "bn": 1e9, "b": 1e9, "billion": 1e9,
    "tn": 1e12, "trillion": 1e12,
}
CURRENCIES = {"$": "USD", "usd": "USD", "dollars": "USD", "dollar": "USD",
              "£": "GBP", "gbp": "GBP", "pounds": "GBP",
              "€": "EUR", "eur": "EUR", "euros": "EUR"}
UNITS = {"%": "percent", "percent": "percent", "per cent": "percent", "pts": "points", "points": "points",
         "kg": "kg", "tonnes": "tonne", "t": "tonne", "km": "km", "miles": "mile"}
CONVERT = {("tonne", "kg"): 1000.0, ("mile", "km"): 1.609344}

_FIG = re.compile(
    r"(?P<cur>[$£€])?\s?(?P<num>-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|-?\d+(?:\.\d+)?)"
    r"\s?(?P<scale>thousand|million|billion|trillion|bn|mn|tn|[kmb](?![a-z]))?"
    r"\s?(?P<unit>per cent|percent|%|pts|points|kg|tonnes|km|miles|t(?![a-z])|usd|gbp|eur|dollars?|pounds|euros)?",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Quantity:
    value: float  # in base units: scale applied
    unit: str | None  # USD, GBP, EUR, percent, points, kg, ...
    text: str
    start: int
    end: int

    def to(self, unit: str) -> Quantity:
        if unit == self.unit:
            return self
        factor = CONVERT.get((self.unit or "", unit))
        if factor is None:
            inverse = CONVERT.get((unit, self.unit or ""))
            if inverse is None:
                raise ValueError(f"cannot convert {self.unit} to {unit}")
            factor = 1 / inverse
        return Quantity(self.value * factor, unit, self.text, self.start, self.end)


def _is_year(num: str, scale: str | None, unit: str | None, cur: str | None) -> bool:
    return not (scale or unit or cur) and re.fullmatch(r"(19|20)\d\d", num) is not None


def parse(text: str) -> list[Quantity]:
    """All figures in ``text``. Bare years are skipped; they are dates, not quantities."""
    out = []
    for m in _FIG.finditer(text):
        num, scale, unit, cur = m.group("num"), m.group("scale"), m.group("unit"), m.group("cur")
        if _is_year(num, scale, unit, cur):
            continue
        value = float(num.replace(",", "")) * (SCALES[scale.lower()] if scale else 1.0)
        code = CURRENCIES.get(cur or "") or CURRENCIES.get((unit or "").lower()) or UNITS.get((unit or "").lower())
        out.append(Quantity(value, code, m.group().strip(), m.start(), m.end()))
    return out


def mixed_currencies(quantities: list[Quantity]) -> bool:
    return len({q.unit for q in quantities if q.unit in {"USD", "GBP", "EUR"}}) > 1


def close(a: float, b: float, tolerance: float = 0.005) -> bool:
    """Relative difference within ``tolerance`` (0.005 is 0.5%)."""
    if a == b:
        return True
    return abs(a - b) / max(abs(a), abs(b)) <= tolerance
