import pytest

from cite_check.nli import LexicalNLI
from cite_check.units import Quantity, close, mixed_currencies, parse, same


@pytest.mark.parametrize("text,value,unit", [
    ("$4.2bn", 4.2e9, "USD"),
    ("4.2 billion dollars", 4.2e9, "USD"),
    ("4,200 million dollars", 4.2e9, "USD"),
    ("£310m", 3.1e8, "GBP"),
    ("18.4%", 18.4, "percent"),
    ("6 per cent", 6.0, "percent"),
    ("12 kg", 12.0, "kg"),
])
def test_parse(text, value, unit):
    q = parse(text)[0]
    assert q.value == pytest.approx(value)
    assert q.unit == unit


def test_bn_and_billion_are_the_same_scale():
    assert parse("$4.2bn")[0].value == parse("$4.2 billion")[0].value


def test_years_are_not_figures():
    assert [q.value for q in parse("Revenue in 2024 was 7 m")] == [7e6]


def test_mixed_currencies():
    assert mixed_currencies(parse("£310m, about $395m"))
    assert not mixed_currencies(parse("$310m and $395m"))


def test_convert():
    assert Quantity(2, "tonne", "2 t", 0, 3).to("kg").value == 2000
    assert Quantity(1609.344, "km", "", 0, 0).to("mile").value == pytest.approx(1000)
    with pytest.raises(ValueError):
        Quantity(1, "kg", "", 0, 0).to("km")


def test_close():
    assert close(100, 100.4)
    assert not close(100, 101)


def test_same_quantity_converts_compatible_units():
    tonnes = parse("1.8 million tonnes")[0]
    assert same(tonnes, parse("1.8 billion kg")[0])
    assert same(parse("5 miles")[0], parse("8.05 km")[0])
    assert not same(parse("12 kg")[0], parse("12 km")[0])  # mass is never length
    assert not same(parse("$12m")[0], parse("£12m")[0])  # currencies are not converted


def test_more_physical_units_parse():
    assert parse("3.5 litres")[0].unit == "liter"
    assert same(parse("3,500 ml")[0], parse("3.5 litres")[0])
    assert same(parse("2.4 GWh")[0], parse("2,400 MWh")[0])


def test_support_check_reads_converted_figures():
    passage = "Freight volumes rose 6% to 1.8 million tonnes."
    judgement = LexicalNLI().judge("Freight volumes rose 6% to 1,800 million kg.", passage)
    assert judgement.label == "supported"
    assert LexicalNLI().judge("Freight volumes rose 6% to 1.8 million km.", passage).label == "contradicted"
