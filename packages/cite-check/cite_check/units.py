"""Figures with their scale and unit: "$4.2bn", "4,200 million dollars", "18.4%", "12 kg".

Physical units are converted through pint, so "1.8 million tonnes" and "1,800 million kg" are the same
figure and "12 kg" is never compared with "12 km". Currencies are never converted: an exchange rate is a
claim of its own.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

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
         "kg": "kg", "g": "g", "tonnes": "tonne", "t": "tonne", "km": "km", "miles": "mile",
         "litres": "liter", "liters": "liter", "litre": "liter", "liter": "liter", "ml": "ml",
         "kwh": "kWh", "mwh": "MWh", "gwh": "GWh", "twh": "TWh"}
# Our unit codes, as pint names. Only these are converted; everything else must match exactly.
PINT = {"kg": "kilogram", "g": "gram", "tonne": "metric_ton", "km": "kilometer", "mile": "mile",
        "liter": "liter", "ml": "milliliter", "kWh": "kilowatt_hour", "MWh": "megawatt_hour",
        "GWh": "gigawatt_hour", "TWh": "terawatt_hour"}

_FIG = re.compile(
    r"(?P<cur>[$£€])?\s?(?P<num>-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|-?\d+(?:\.\d+)?)"
    r"\s?(?P<scale>thousand|million|billion|trillion|bn|mn|tn|[kmb](?![a-z]))?"
    r"\s?(?P<unit>per cent|percent|%|pts|points|kwh|mwh|gwh|twh|kg|tonnes|km|miles|litres?|liters?|ml"
    r"|t(?![a-z])|g(?![a-z])|usd|gbp|eur|dollars?|pounds|euros)?",
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
        """The same quantity in ``unit``. Raises ValueError for units of another dimension or no unit."""
        if unit == self.unit:
            return self
        if self.unit not in PINT or unit not in PINT:
            raise ValueError(f"cannot convert {self.unit} to {unit}")
        import pint

        try:
            value = _registry().Quantity(self.value, PINT[self.unit]).to(PINT[unit]).magnitude
        except pint.DimensionalityError as exc:
            raise ValueError(f"cannot convert {self.unit} to {unit}") from exc
        return Quantity(float(value), unit, self.text, self.start, self.end)


@lru_cache(maxsize=1)
def _registry():
    import pint

    return pint.UnitRegistry()


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


def same(a: Quantity, b: Quantity, tolerance: float = 0.005) -> bool:
    """True when two figures state the same quantity, converting physical units first.

    A figure without a unit is compared by value, as a claim often drops the unit its source states.
    Two different units that cannot be converted (currencies, or kg against km) are never the same.
    """
    if a.unit is None or b.unit is None or a.unit == b.unit:
        return close(a.value, b.value, tolerance)
    try:
        return close(a.value, b.to(a.unit).value, tolerance)
    except ValueError:
        return False


def close(a: float, b: float, tolerance: float = 0.005) -> bool:
    """Relative difference within ``tolerance`` (0.005 is 0.5%)."""
    if a == b:
        return True
    return abs(a - b) / max(abs(a), abs(b)) <= tolerance
