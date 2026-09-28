"""Canonical body/telemetry contract and nominal spatial conversions."""
from datetime import datetime, timezone
import json
import math

import pytest

from home_cortex.spatial.embodiment import (
    Embodiment, LocalFrame, embodiment_as_mapping,
    embodiment_from_mapping,
)
from home_cortex.spatial.primitives import SpatialContractError
from home_cortex.spatial.telemetry import (
    PhysicalValue, box_corners_in_space, embodiment_point_in_physical,
    embodiment_point_in_space, estimate_within_limits,
    physical_telemetry_as_mapping, physical_telemetry_from_mapping,
)
from home_cortex.spatial.transforms import Position, canonical_basis


def _body() -> dict:
    return {
        "id": "embodiment:microduck-01", "name": "MicroDuck", "agent_id": "agent:butler",
        "local_frame": {"forward": "+x", "left": "+y", "up": "+z"},
        "geometry": {"box": {"length_m": 0.32, "width_m": 0.24, "height_m": 0.18,
                             "center": {"x": 0.04, "y": 0.0, "z": 0.09}}},
    }


def _telemetry(*, validity="valid") -> dict:
    return {
        "embodiment_id": "embodiment:microduck-01", "space_id": "space:kitchen",
        "measured_at": "2026-09-27T12:00:00-07:00", "validity": validity,
        "transform": {name: {"value": value, "p95": uncertainty} for name, value, uncertainty in (
            ("x", 1.0, 0.02), ("y", 2.0, 0.02), ("z", 0.0, 0.03),
            ("yaw", math.pi / 2, 0.04), ("pitch", 0.0, 0.01), ("roll", 0.0, 0.01),
        )} if validity == "valid" else None,
    }


def test_valid_body_and_round_trip() -> None:
    original = embodiment_from_mapping(_body())
    assert isinstance(original, Embodiment)
    assert original.geometry.center == Position(0.04, 0, 0.09)
    assert original.local_frame.axis("forward") == Position(1, 0, 0)
    assert embodiment_from_mapping(embodiment_as_mapping(original)) == original
    assert embodiment_from_mapping(json.loads(json.dumps(embodiment_as_mapping(original)))) == original


@pytest.mark.parametrize("dimension", [0, -0.2, math.nan, math.inf, True])
def test_invalid_box_dimensions(dimension) -> None:
    payload = _body()
    payload["geometry"]["box"]["length_m"] = dimension
    with pytest.raises(SpatialContractError):
        embodiment_from_mapping(payload)


def test_body_center_and_axes_must_be_explicit_and_right_handed() -> None:
    payload = _body()
    del payload["geometry"]["box"]["center"]
    with pytest.raises(SpatialContractError):
        embodiment_from_mapping(payload)
    payload = _body()
    payload["local_frame"]["left"] = "-y"
    with pytest.raises(SpatialContractError, match="forward × left"):
        embodiment_from_mapping(payload)
    assert LocalFrame("+y", "-x", "+z").axis("forward") == Position(0, 1, 0)


def test_valid_telemetry_timestamp_and_round_trip() -> None:
    original = physical_telemetry_from_mapping(_telemetry())
    assert original.measured_at == datetime(2026, 9, 27, 19, tzinfo=timezone.utc)
    assert original.transform is not None
    assert original.transform.nominal_pose().position == Position(1, 2, 0)
    assert physical_telemetry_from_mapping(physical_telemetry_as_mapping(original)) == original
    assert physical_telemetry_from_mapping(json.loads(json.dumps(physical_telemetry_as_mapping(original)))) == original


@pytest.mark.parametrize("bad", [-0.01, math.nan, math.inf, True])
def test_bad_uncertainty_rejected(bad) -> None:
    payload = _telemetry()
    payload["transform"]["x"]["p95"] = bad
    with pytest.raises(SpatialContractError):
        physical_telemetry_from_mapping(payload)
    with pytest.raises(SpatialContractError):
        PhysicalValue(1.0, bad)


def test_missing_estimate_is_explicit_and_space_may_be_unknown() -> None:
    payload = _telemetry(validity="no_estimate")
    payload["space_id"] = None
    value = physical_telemetry_from_mapping(payload)
    assert value.transform is None
    assert physical_telemetry_as_mapping(value) == {
        **payload, "measured_at": "2026-09-27T19:00:00+00:00"}
    payload["transform"] = _telemetry()["transform"]
    with pytest.raises(SpatialContractError, match="cannot contain"):
        physical_telemetry_from_mapping(payload)
    payload = _telemetry()
    payload["space_id"] = None
    with pytest.raises(SpatialContractError, match="requires space_id"):
        physical_telemetry_from_mapping(payload)


@pytest.mark.parametrize("stamp", ["2026-09-27T12:00:00", "not a timestamp", None])
def test_timestamp_requires_measurement_time_with_offset(stamp) -> None:
    payload = _telemetry()
    payload["measured_at"] = stamp
    with pytest.raises(SpatialContractError, match="measured_at"):
        physical_telemetry_from_mapping(payload)


def test_quality_is_derived_from_validity_uncertainty_and_age() -> None:
    stamp = datetime(2026, 9, 27, 19, 0, 1, tzinfo=timezone.utc)
    limits = dict(now=stamp, max_age_s=2, max_position_p95_m=0.03,
                  max_orientation_p95_rad=0.05)
    assert estimate_within_limits(physical_telemetry_from_mapping(_telemetry()), **limits)
    assert not estimate_within_limits(physical_telemetry_from_mapping(_telemetry(validity="no_estimate")), **limits)
    assert not estimate_within_limits(physical_telemetry_from_mapping(_telemetry()),
                                      **{**limits, "max_age_s": 0.5})
    assert not estimate_within_limits(physical_telemetry_from_mapping(_telemetry()),
                                      **{**limits, "max_position_p95_m": 0.01})


def test_local_to_space_and_rotated_offset_box() -> None:
    body = embodiment_from_mapping(_body())
    telemetry = physical_telemetry_from_mapping(_telemetry())
    assert telemetry.transform is not None
    point = embodiment_point_in_space(Position(0.04, 0, 0.09), telemetry.transform)
    assert (point.x, point.y, point.z) == pytest.approx((1.0, 2.04, 0.09))
    corners = box_corners_in_space(body.geometry, body.local_frame, telemetry.transform)
    assert len(corners) == 8
    assert sorted({round(p.x, 9) for p in corners}) == [0.88, 1.12]
    assert sorted({round(p.y, 9) for p in corners}) == [1.88, 2.2]
    assert sorted({round(p.z, 9) for p in corners}) == [0, 0.18]


def test_box_respects_configured_intrinsic_axes() -> None:
    body = embodiment_from_mapping(_body())
    telemetry_payload = _telemetry()
    telemetry_payload["transform"]["yaw"]["value"] = 0
    transform = physical_telemetry_from_mapping(telemetry_payload).transform
    assert transform is not None
    corners = box_corners_in_space(body.geometry, LocalFrame("+y", "-x", "+z"), transform)
    assert sorted({round(p.x, 9) for p in corners}) == [0.92, 1.16]
    assert sorted({round(p.y, 9) for p in corners}) == [1.84, 2.16]


def test_scaled_and_rotated_space_basis_preserves_physical_box_size() -> None:
    basis = canonical_basis({"x": [0, 2, 0], "y": [-3, 0, 0], "z": [0, 0, 0.5]})
    body = embodiment_from_mapping(_body())
    payload = _telemetry()
    payload["transform"]["yaw"]["value"] = 0
    telemetry = physical_telemetry_from_mapping(payload)
    assert telemetry.transform is not None
    corners = box_corners_in_space(body.geometry, body.local_frame, telemetry.transform,
                                   space_basis=basis)
    physical = [embodiment_point_in_physical(Position(x, y, z), telemetry.transform,
                                            space_basis=basis)
                for x, y, z in ((-0.12, -0.12, 0), (0.2, 0.12, 0.18))]
    assert physical[1].x - physical[0].x == pytest.approx(-0.24)
    assert physical[1].y - physical[0].y == pytest.approx(0.32)
    assert physical[1].z - physical[0].z == pytest.approx(0.18)
    assert corners[0].x == pytest.approx(0.94)
    assert corners[0].y == pytest.approx(1.96)
    assert not estimate_within_limits(
        telemetry, now=datetime(2026, 9, 27, 19, 0, 1, tzinfo=timezone.utc),
        max_age_s=2, max_position_p95_m=0.05, max_orientation_p95_rad=0.05,
        space_basis=basis,
    )


def test_sheared_basis_cannot_define_rigid_body_orientation() -> None:
    basis = canonical_basis({"x": [1, 0, 0], "y": [1, 1, 0], "z": [0, 0, 1]})
    telemetry = physical_telemetry_from_mapping(_telemetry())
    assert telemetry.transform is not None
    with pytest.raises(SpatialContractError, match="orthogonal"):
        embodiment_point_in_space(Position(), telemetry.transform, space_basis=basis)
