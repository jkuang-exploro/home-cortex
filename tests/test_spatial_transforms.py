"""Pure nested-space pose math; no database or TF tree."""
import math

import pytest

from home_cortex.spatial.transforms import (
    IDENTITY_BASIS,
    Pose,
    Position,
    SpatialTransformError,
    canonical_basis,
    compose_chain,
    compose_pose,
    invert_pose,
    local_to_physical,
    physical_to_local,
    pose,
    pose_as_mapping,
    pose_from_mapping,
    transform_point,
    transform_pose,
)
from home_cortex.spatial.units import normalize_angle, normalize_length

ABS = 1e-9
RIGHT = normalize_angle(90, "deg")


def _approx_position(actual: Position, expected: tuple[float, float, float]) -> None:
    assert actual.x == pytest.approx(expected[0], abs=ABS)
    assert actual.y == pytest.approx(expected[1], abs=ABS)
    assert actual.z == pytest.approx(expected[2], abs=ABS)


def _approx_pose(actual: Pose, expected: Pose) -> None:
    _approx_position(actual.position, expected.position.as_tuple())
    assert actual.orientation.yaw == pytest.approx(expected.orientation.yaw, abs=ABS)
    assert actual.orientation.pitch == pytest.approx(expected.orientation.pitch, abs=ABS)
    assert actual.orientation.roll == pytest.approx(expected.orientation.roll, abs=ABS)


def test_translation_only() -> None:
    frame = pose(x=1.0, y=2.0, z=3.0)
    _approx_position(transform_point((0.0, 0.0, 0.0), frame), (1.0, 2.0, 3.0))
    _approx_position(transform_point((0.5, -0.25, 0.1), frame), (1.5, 1.75, 3.1))


def test_yaw_rotation() -> None:
    frame = pose(yaw=RIGHT)
    _approx_position(transform_point((1.0, 0.0, 0.0), frame), (0.0, 1.0, 0.0))
    _approx_position(transform_point((0.0, 1.0, 0.0), frame), (-1.0, 0.0, 0.0))


def test_combined_translation_and_rotation() -> None:
    frame = pose(x=2.0, y=0.5, z=0.0, yaw=RIGHT)
    _approx_position(transform_point((0.3, 0.0, 1.2), frame), (2.0, 0.8, 1.2))
    child = pose(x=0.3, z=1.2)
    moved = transform_pose(child, frame)
    _approx_position(moved.position, (2.0, 0.8, 1.2))
    assert moved.orientation.yaw == pytest.approx(RIGHT, abs=ABS)


def test_inverse_transform_round_trips() -> None:
    frame = pose(x=1.2, y=-0.4, z=0.8, yaw=0.7, pitch=-0.2, roll=0.15)
    local = (0.25, -0.1, 0.05)
    world = transform_point(local, frame)
    recovered = transform_point(world, invert_pose(frame))
    _approx_position(recovered, local)
    _approx_pose(compose_pose(frame, invert_pose(frame)), Pose())


def test_nested_two_level_transform() -> None:
    fridge_in_kitchen = pose(x=2.0, y=0.5)
    milk_in_fridge = pose(x=0.1, y=0.05, z=1.2)
    milk_in_kitchen = compose_pose(fridge_in_kitchen, milk_in_fridge)
    _approx_position(milk_in_kitchen.position, (2.1, 0.55, 1.2))


def test_nested_three_level_kitchen_hierarchy() -> None:
    fridge_in_kitchen = pose(
        x=normalize_length(200, "cm"),
        y=normalize_length(50, "cm"),
        yaw=RIGHT,
    )
    shelf_in_fridge = pose(x=0.3, z=1.2)
    milk_in_shelf = pose(x=0.1, y=0.05, z=0.02)
    milk_in_kitchen = compose_chain(
        (fridge_in_kitchen, shelf_in_fridge, milk_in_shelf)
    )
    # Fridge yaw +90 maps shelf +x to kitchen +y.
    _approx_position(milk_in_kitchen.position, (1.95, 0.9, 1.22))
    milk_point = transform_point((0.0, 0.0, 0.0), milk_in_kitchen)
    _approx_position(milk_point, (1.95, 0.9, 1.22))


def test_child_to_ancestor_and_ancestor_to_child() -> None:
    fridge_in_kitchen = pose(x=2.0, y=0.5, yaw=RIGHT)
    shelf_in_fridge = pose(x=0.3, z=1.2)
    shelf_in_kitchen = compose_pose(fridge_in_kitchen, shelf_in_fridge)
    local = (0.1, 0.05, 0.02)
    in_kitchen = transform_point(local, shelf_in_kitchen)
    back_to_shelf = transform_point(in_kitchen, invert_pose(shelf_in_kitchen))
    _approx_position(back_to_shelf, local)
    in_fridge = transform_point(in_kitchen, invert_pose(fridge_in_kitchen))
    _approx_position(in_fridge, transform_point(local, shelf_in_fridge).as_tuple())


def test_same_physical_point_in_each_ancestor_frame() -> None:
    fridge_in_kitchen = pose(x=2.0, y=0.5)
    shelf_in_fridge = pose(z=1.2)
    milk_in_shelf = pose(x=0.1, y=0.05, z=0.02)
    in_shelf = milk_in_shelf.position
    in_fridge = transform_point(in_shelf, shelf_in_fridge)
    in_kitchen = transform_point(in_fridge, fridge_in_kitchen)
    _approx_position(in_fridge, (0.1, 0.05, 1.22))
    _approx_position(in_kitchen, (2.1, 0.55, 1.22))
    via_chain = compose_chain((fridge_in_kitchen, shelf_in_fridge, milk_in_shelf))
    _approx_position(via_chain.position, in_kitchen.as_tuple())


def test_disconnected_transform_request_fails() -> None:
    with pytest.raises(SpatialTransformError, match="disconnected transform"):
        compose_chain((pose(x=1.0), None, pose(x=0.1)))
    with pytest.raises(SpatialTransformError, match="finite"):
        pose(x=math.inf)
    with pytest.raises(SpatialTransformError, match="Unknown pose fields"):
        pose_from_mapping({"position": {"x": 0, "y": 0, "z": 0}, "frame_id": "map"})


def test_contract_mapping_round_trip() -> None:
    original = pose(x=1.42, y=0.83, z=0.91, yaw=0.2)
    restored = pose_from_mapping(pose_as_mapping(original))
    _approx_pose(restored, original)


def test_empty_chain_is_identity() -> None:
    _approx_pose(compose_chain(()), Pose())


def test_identity_basis_preserves_local_coordinates() -> None:
    point = (1.37, 0.5, 0.31)
    _approx_position(local_to_physical(point, IDENTITY_BASIS), point)
    _approx_position(physical_to_local(point, IDENTITY_BASIS), point)


def test_scaled_z_basis_maps_local_height_to_physical_meters() -> None:
    basis = canonical_basis({
        "x": [1.0, 0.0, 0.0],
        "y": [0.0, 1.0, 0.0],
        "z": [0.0, 0.0, 0.1],
    })
    _approx_position(local_to_physical((0.0, 0.0, 1.0), basis), (0.0, 0.0, 0.1))
    _approx_position(local_to_physical((0.0, 0.0, 0.5), basis), (0.0, 0.0, 0.05))
    _approx_position(physical_to_local((0.0, 0.0, 0.1), basis), (0.0, 0.0, 1.0))
    _approx_position(physical_to_local((0.0, 0.0, 0.05), basis), (0.0, 0.0, 0.5))


def test_rotated_and_skewed_bases_are_invertible() -> None:
    rotated = canonical_basis({
        "x": [0.0, 1.0, 0.0],
        "y": [-1.0, 0.0, 0.0],
        "z": [0.0, 0.0, 1.0],
    })
    _approx_position(local_to_physical((1.0, 0.0, 0.0), rotated), (0.0, 1.0, 0.0))
    _approx_position(physical_to_local((0.0, 1.0, 0.0), rotated), (1.0, 0.0, 0.0))
    skewed = canonical_basis({
        "x": [1.0, 0.0, 0.0],
        "y": [1.0, 1.0, 0.0],
        "z": [0.0, 0.0, 1.0],
    })
    physical = local_to_physical((2.0, 3.0, 0.5), skewed)
    _approx_position(physical, (5.0, 3.0, 0.5))
    _approx_position(physical_to_local(physical, skewed), (2.0, 3.0, 0.5))


def test_singular_basis_is_rejected() -> None:
    with pytest.raises(SpatialTransformError, match="zero vector"):
        canonical_basis({
            "x": [1.0, 0.0, 0.0],
            "y": [0.0, 1.0, 0.0],
            "z": [0.0, 0.0, 0.0],
        })
    with pytest.raises(SpatialTransformError, match="linearly independent"):
        canonical_basis({
            "x": [1.0, 0.0, 0.0],
            "y": [2.0, 0.0, 0.0],
            "z": [0.0, 0.0, 1.0],
        })
    with pytest.raises(SpatialTransformError, match="finite"):
        canonical_basis({
            "x": [math.inf, 0.0, 0.0],
            "y": [0.0, 1.0, 0.0],
            "z": [0.0, 0.0, 1.0],
        })
