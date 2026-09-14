"""Fiducial observation contract; no detector."""
import math

import pytest

from home_cortex.spatial.contracts import SpatialContractError
from home_cortex.spatial.observation import parse_fiducial_observation, parse_fiducial_observations
from home_cortex.spatial.units import as_meters, normalize_angle

ABS = 1e-12


def _observation(**fields):
    payload = {
        "anchor": "anchor:kitchen_x_137",
        "position": {"x": 0.4, "y": 0.0, "z": 0.2},
        "orientation": {"yaw": 0.0, "pitch": 0.0, "roll": 0.0},
        "timestamp": "2026-09-13T12:00:00-07:00",
        "confidence": 0.92,
        "reprojection_error_px": 0.8,
    }
    payload.update(fields)
    return payload


def test_valid_fiducial_observation() -> None:
    observation = parse_fiducial_observation(_observation())
    assert observation.anchor_id == "anchor:kitchen_x_137"
    assert observation.pose.position.x == pytest.approx(0.4, abs=ABS)
    assert observation.confidence == pytest.approx(0.92, abs=ABS)
    assert observation.reprojection_error_px == pytest.approx(0.8, abs=ABS)


def test_observation_normalizes_structured_units() -> None:
    observation = parse_fiducial_observation(_observation(
        position={
            "x": {"value": 40, "unit": "cm"},
            "y": 0.0,
            "z": {"value": 8, "unit": "in"},
        },
        orientation={
            "yaw": {"value": 90, "unit": "deg"},
            "pitch": 0.0,
            "roll": 0.0,
        },
    ))
    assert observation.pose.position.x == pytest.approx(as_meters({"value": 40, "unit": "cm"}), abs=ABS)
    assert observation.pose.orientation.yaw == pytest.approx(normalize_angle(90, "deg"), abs=ABS)


def test_empty_observation_list_is_valid() -> None:
    assert parse_fiducial_observations(None) == ()
    assert parse_fiducial_observations([]) == ()


@pytest.mark.parametrize(
    "mutate",
    [
        lambda p: p.update(confidence=1.2),
        lambda p: p.update(reprojection_error_px=-0.1),
        lambda p: p["position"].update(x=math.nan),
        lambda p: p.update(timestamp="later"),
        lambda p: p.update(anchor="item:tag"),
    ],
)
def test_invalid_observation_is_rejected(mutate) -> None:
    payload = _observation()
    mutate(payload)
    with pytest.raises(SpatialContractError):
        parse_fiducial_observation(payload)
