"""Optional surveyed anchors; spaces stay valid without them."""
import json
from pathlib import Path

import pytest

from home_cortex.spatial.anchors import (
    parse_surveyed_anchor,
    parse_surveyed_anchors,
    surveyed_anchor_as_mapping,
)
from home_cortex.spatial.contracts import SpatialContractError, apply_space_spatial_fields
from home_cortex.spatial.pose import parse_runtime_pose
from home_cortex.spatial.units import as_meters

ROOT = Path(__file__).parents[1]
ABS = 1e-12


def _anchor(suffix: str, x: float, y: float, z: float = 0.31, **fields):
    payload = {
        "id": f"anchor:kitchen_{suffix}",
        "space": "space:kitchen",
        "type": "fiducial",
        "position": {"x": x, "y": y, "z": z},
        "survey": {"method": "tape_measure", "uncertainty_m": 0.005},
    }
    payload.update(fields)
    return payload


def test_one_surveyed_anchor() -> None:
    anchor = parse_surveyed_anchor(_anchor("x_137", 1.37, 0.0))
    assert anchor.id == "anchor:kitchen_x_137"
    assert anchor.space == "space:kitchen"
    assert anchor.kind == "fiducial"
    assert anchor.position.x == pytest.approx(1.37, abs=ABS)
    assert anchor.position.y == pytest.approx(0.0, abs=ABS)
    assert anchor.survey is not None
    assert anchor.survey.method == "tape_measure"


def test_multiple_anchors_including_same_axis() -> None:
    anchors = parse_surveyed_anchors([
        _anchor("x_137", 1.37, 0.0),
        _anchor("x_100", 1.00, 0.0),
        _anchor("y_105", 0.0, 1.05),
        _anchor("y_200", 0.0, 2.00),
    ])
    assert len(anchors) == 4
    xs = [item.position.x for item in anchors if item.position.y == 0.0]
    ys = [item.position.y for item in anchors if item.position.x == 0.0]
    assert xs == pytest.approx([1.37, 1.00], abs=ABS)
    assert ys == pytest.approx([1.05, 2.00], abs=ABS)


def test_anchors_need_not_be_origin_or_room_limits() -> None:
    anchor = parse_surveyed_anchor(_anchor(
        "interior",
        1.37,
        1.05,
        recognition={"family": "apriltag", "marker_id": 17},
        orientation={"yaw": 1.5708, "pitch": 0.0, "roll": 0.0},
    ))
    assert anchor.position.x not in {0.0}
    assert anchor.position.y not in {0.0}
    assert anchor.recognition is not None
    assert anchor.recognition.family == "apriltag"
    assert anchor.recognition.marker_id == 17


def test_anchor_units_normalize_from_cm_and_inches() -> None:
    anchor = parse_surveyed_anchor(_anchor(
        "x_137",
        {"value": 137, "unit": "cm"},
        0.0,
        z={"value": 12.2, "unit": "in"},
    ))
    assert anchor.position.x == pytest.approx(1.37, abs=ABS)
    assert anchor.position.z == pytest.approx(as_meters({"value": 12.2, "unit": "in"}), abs=ABS)


def test_anchor_removal_leaves_space_valid() -> None:
    space = {
        "id": "space:kitchen",
        "name": {"en": "Kitchen"},
        "coordinate": {"unit": "m", "origin": [0.0, 0.0, 0.0], "z_axis": "up"},
        "geometry": {"type": "box", "size": [4.8, 3.6, 2.7]},
        "navigable": True,
        "accessible": True,
    }
    apply_space_spatial_fields(space, source="space.json")
    before = dict(space)
    assert parse_surveyed_anchors([_anchor("x_137", 1.37, 0.0)])
    assert parse_surveyed_anchors([]) == ()
    assert parse_surveyed_anchors(None) == ()
    apply_space_spatial_fields(space, source="space.json")
    assert space == before


def test_legacy_spaces_remain_valid_without_pose_or_anchors() -> None:
    space = {"id": "space:kitchen", "space_type": "room", "name": {"en": "Kitchen"}}
    apply_space_spatial_fields(space, source="space.json")
    assert parse_runtime_pose(None) is None
    assert parse_surveyed_anchors(None) == ()
    production = ROOT / "data" / "nodes" / "space.json"
    for record in json.loads(production.read_text()):
        apply_space_spatial_fields(record, source=str(production))
        assert "coordinate" not in record or record["coordinate"]["unit"] == "m"


def test_duplicate_anchor_ids_are_rejected() -> None:
    with pytest.raises(SpatialContractError, match="Duplicate"):
        parse_surveyed_anchors([
            _anchor("x_137", 1.37, 0.0),
            _anchor("x_137", 1.00, 0.0),
        ])


def test_anchor_mapping_round_trip() -> None:
    original = parse_surveyed_anchor(_anchor(
        "x_137",
        1.37,
        0.0,
        recognition={"family": "apriltag", "marker_id": 17},
    ))
    assert parse_surveyed_anchor(surveyed_anchor_as_mapping(original)) == original


def test_invalid_anchor_id_table_is_rejected() -> None:
    with pytest.raises(SpatialContractError, match="anchor:"):
        parse_surveyed_anchor(_anchor("x_137", 1.37, 0.0) | {"id": "item:marker"})
