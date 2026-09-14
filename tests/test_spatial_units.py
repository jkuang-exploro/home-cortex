"""Structured unit conversion; no natural-language parsing."""
import math

import pytest

from home_cortex.spatial.units import (
    SpatialUnitError,
    format_angle,
    format_length,
    normalize_angle,
    normalize_length,
)

ABS = 1e-12


def test_millimeters_normalize_to_meters() -> None:
    assert normalize_length(1370, "mm") == pytest.approx(1.37, abs=ABS)


def test_centimeters_normalize_to_meters() -> None:
    assert normalize_length(137, "cm") == pytest.approx(1.37, abs=ABS)
    assert normalize_length(105, "cm") == pytest.approx(1.05, abs=ABS)


def test_meters_normalize_to_meters() -> None:
    assert normalize_length(1.37, "m") == pytest.approx(1.37, abs=ABS)


def test_inches_normalize_to_meters() -> None:
    assert normalize_length(54, "in") == pytest.approx(1.3716, abs=ABS)


def test_feet_normalize_to_meters() -> None:
    assert normalize_length(4, "ft") == pytest.approx(1.2192, abs=ABS)


def test_structured_feet_and_inches_normalize_to_meters() -> None:
    assert normalize_length({"ft": 4, "in": 6}) == pytest.approx(1.3716, abs=ABS)


def test_degrees_normalize_to_radians() -> None:
    assert normalize_angle(180, "deg") == pytest.approx(math.pi, abs=ABS)
    assert normalize_angle(90, "degrees") == pytest.approx(math.pi / 2, abs=ABS)
    assert normalize_angle(math.pi, "rad") == pytest.approx(math.pi, abs=ABS)


def test_metric_rendering() -> None:
    meters = normalize_length(137, "cm")
    assert format_length(meters, unit="m") == "1.37 m"
    assert format_length(meters, unit="cm") == "137 cm"
    assert format_length(meters, unit_system="metric") == "1.37 m"


def test_imperial_rendering() -> None:
    meters = normalize_length(137, "cm")
    assert format_length(meters, unit="in") == "53.94 in"
    assert format_length(meters, unit_system="imperial") == "4 ft 5.94 in"
    assert format_length(normalize_length(4, "ft"), unit_system="imperial") == "4 ft"


def test_length_round_trip_within_tolerance() -> None:
    original = 54
    meters = normalize_length(original, "in")
    rendered = format_length(meters, unit="in")
    amount, unit = rendered.rsplit(" ", 1)
    assert unit == "in"
    assert normalize_length(float(amount), unit) == pytest.approx(meters, abs=1e-4)


def test_angle_round_trip_within_tolerance() -> None:
    radians = normalize_angle(45, "deg")
    rendered = format_angle(radians, unit="deg")
    amount, unit = rendered.rsplit(" ", 1)
    assert unit == "deg"
    assert normalize_angle(float(amount), unit) == pytest.approx(radians, abs=1e-8)


@pytest.mark.parametrize(
    "value, unit",
    [(math.inf, "m"), (math.nan, "cm"), ("1.37", "m"), (1.37, "yard")],
)
def test_invalid_length_is_rejected(value, unit) -> None:
    with pytest.raises(SpatialUnitError):
        normalize_length(value, unit)
