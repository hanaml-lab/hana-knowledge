import pytest

from cite_check.units import Quantity, close, mixed_currencies, parse


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
