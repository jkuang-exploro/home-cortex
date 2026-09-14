"""Runtime robot pose contract; no localization algorithm."""
import math

import pytest

from home_cortex.spatial.contracts import SpatialContractError
from home_cortex.spatial.pose import (
    LOCALIZATION_QUALITY,
    parse_runtime_pose,
    runtime_pose_as_mapping,
)
from home_cortex.spatial.units import as_meters, normalize_angle

ABS = 1e-12


def _localized(**fields):
    payload = {
        "agent": "agent:microduck",
        "space": "space:kitchen",
        "position": {"x": 1.372, "y": 2.048, "z": 0.0},
        "orientation": {"yaw": 1.571, "pitch": 0.0, "roll": 0.0},
        "timestamp": "2026-09-13T12:00:00-07:00",
        "source": "external_localization",
        "quality": "localized",
        "uncertainty": {"x_sigma": 0.012, "y_sigma": 0.015, "z_sigma": 0.02, "yaw_sigma": 0.018},
    }
    payload.update(fields)
    return payload


def test_valid_robot_pose() -> None:
    pose = parse_runtime_pose(_localized())
    assert pose is not None
    assert pose.agent == "agent:microduck"
    assert pose.space == "space:kitchen"
    assert pose.quality == "localized"
    assert pose.source == "external_localization"
    assert pose.timestamp == "2026-09-13T12:00:00-07:00"
    assert pose.estimate is not None
    assert pose.estimate.position.x == pytest.approx(1.372, abs=ABS)
    assert pose.uncertainty is not None
    assert pose.uncertainty.x_sigma == pytest.approx(0.012, abs=ABS)


def test_arbitrary_start_coordinates_and_yaw() -> None:
    pose = parse_runtime_pose(_localized(
        position={"x": 3.91, "y": 0.22, "z": 0.0},
        orientation={"yaw": 4.2, "pitch": 0.0, "roll": 0.0},
    ))
    assert pose is not None and pose.estimate is not None
    assert pose.estimate.position.x == pytest.approx(3.91, abs=ABS)
    assert pose.estimate.position.y == pytest.approx(0.22, abs=ABS)
    assert pose.estimate.orientation.yaw == pytest.approx(4.2, abs=ABS)


def test_canonical_si_from_structured_units() -> None:
    pose = parse_runtime_pose(_localized(
        position={
            "x": {"value": 137.2, "unit": "cm"},
            "y": {"value": 80.63, "unit": "in"},
            "z": 0.0,
        },
        orientation={
            "yaw": {"value": 90, "unit": "deg"},
            "pitch": 0.0,
            "roll": 0.0,
        },
    ))
    assert pose is not None and pose.estimate is not None
    assert pose.estimate.position.x == pytest.approx(as_meters({"value": 137.2, "unit": "cm"}), abs=ABS)
    assert pose.estimate.position.y == pytest.approx(as_meters({"value": 80.63, "unit": "in"}), abs=ABS)
    assert pose.estimate.orientation.yaw == pytest.approx(normalize_angle(90, "deg"), abs=ABS)


def test_missing_pose_is_unknown() -> None:
    assert parse_runtime_pose(None) is None
    pose = parse_runtime_pose({
        "agent": "agent:microduck",
        "space": "space:kitchen",
        "quality": "unknown",
    })
    assert pose is not None
    assert pose.estimate is None
    assert pose.quality == "unknown"
    assert "unknown" in LOCALIZATION_QUALITY


def test_uncertainty_and_provenance_are_preserved() -> None:
    pose = parse_runtime_pose(_localized())
    assert pose is not None
    restored = parse_runtime_pose(runtime_pose_as_mapping(pose))
    assert restored == pose
    assert restored is not None
    assert restored.uncertainty is not None
    assert restored.uncertainty.y_sigma == pytest.approx(0.015, abs=ABS)
    assert restored.source == "external_localization"


@pytest.mark.parametrize(
    "mutate",
    [
        lambda p: p["position"].update(x=math.nan),
        lambda p: p["position"].update(y=math.inf),
        lambda p: p["orientation"].update(yaw=math.inf),
        lambda p: p["uncertainty"].update(x_sigma=-0.01),
        lambda p: p.update(timestamp="not-a-time"),
        lambda p: p.update(quality="confident"),
    ],
)
def test_invalid_pose_values_are_rejected(mutate) -> None:
    payload = _localized()
    mutate(payload)
    with pytest.raises(SpatialContractError):
        parse_runtime_pose(payload)


def test_localized_without_coordinates_is_rejected() -> None:
    with pytest.raises(SpatialContractError, match="requires position"):
        parse_runtime_pose({
            "agent": "agent:microduck",
            "space": "space:kitchen",
            "quality": "localized",
        })


def test_unknown_pose_cannot_carry_coordinates() -> None:
    with pytest.raises(SpatialContractError, match="unknown"):
        parse_runtime_pose(_localized(quality="unknown"))
