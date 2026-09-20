"""Arbitrary-start surveyed-anchor localization. Synthetic observations only."""
import json
import math
from pathlib import Path

import pytest

from home_cortex.spatial.localization.anchors import SurveyedAnchor, anchors_from_space
from home_cortex.spatial.contracts import apply_space_spatial_fields
from home_cortex.spatial.localization.solver import (
    LOCALIZED_TRANSLATION_M,
    LOCALIZED_YAW_RAD,
    localize_from_fiducials,
    localize_in_space,
    synthetic_observation,
)
from home_cortex.spatial.localization.observation import FiducialObservation
from home_cortex.spatial.transforms import Pose, pose
from home_cortex.spatial.units import as_meters, normalize_angle

ABS = 1e-9
STAMP = "2026-09-13T12:00:00-07:00"
AGENT = "agent:microduck"
SPACE = "space:kitchen"
RIGHT = normalize_angle(90, "deg")
CAMERA = pose(x=0.08, z=0.12)
STATIC_SPACES = Path(__file__).parent / "static_test_data" / "nodes" / "space.json"


def kitchen_space() -> dict:
    kitchen = next(
        record
        for record in json.loads(STATIC_SPACES.read_text())
        if record["id"] == SPACE
    )
    apply_space_spatial_fields(kitchen, source=str(STATIC_SPACES))
    return kitchen


def kitchen_anchors() -> tuple[SurveyedAnchor, ...]:
    return anchors_from_space(kitchen_space())


def _observe(body: Pose, anchors: tuple[SurveyedAnchor, ...], count: int | None = None) -> tuple:
    selected = anchors if count is None else anchors[:count]
    return tuple(
        synthetic_observation(
            body_in_space=body, camera_in_body=CAMERA, anchor=item, timestamp=STAMP
        )
        for item in selected
    )


def _solve(body: Pose, observations) -> object:
    return localize_from_fiducials(
        agent=AGENT,
        space=SPACE,
        anchors=kitchen_anchors(),
        observations=observations,
        camera_in_body=CAMERA,
        timestamp=STAMP,
    )


def _wrap(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


def _approx_pose(actual: Pose, expected: Pose) -> None:
    assert actual.position.x == pytest.approx(expected.position.x, abs=ABS)
    assert actual.position.y == pytest.approx(expected.position.y, abs=ABS)
    assert actual.position.z == pytest.approx(expected.position.z, abs=ABS)
    assert _wrap(actual.orientation.yaw - expected.orientation.yaw) == pytest.approx(0.0, abs=ABS)
    assert _wrap(actual.orientation.pitch - expected.orientation.pitch) == pytest.approx(0.0, abs=ABS)
    assert _wrap(actual.orientation.roll - expected.orientation.roll) == pytest.approx(0.0, abs=ABS)


@pytest.mark.parametrize(
    "body",
    [
        pose(x=1.2, y=1.8, yaw=0.0),
        pose(x=0.4, y=0.5, yaw=RIGHT),
        pose(x=3.1, y=2.4, yaw=math.pi),
        pose(x=2.2, y=0.7, yaw=-2.1),
        pose(x=0.15, y=3.0, yaw=0.4),
    ],
)
def test_one_anchor_recovers_arbitrary_start(body: Pose) -> None:
    solution = _solve(body, _observe(body, kitchen_anchors(), 1))
    assert solution.pose.quality == "localized"
    assert solution.pose.space == SPACE
    assert solution.pose.agent == AGENT
    assert solution.pose.estimate is not None
    _approx_pose(solution.pose.estimate, body)
    assert solution.translation_rmse_m == pytest.approx(0.0, abs=ABS)
    assert solution.yaw_rmse_rad == pytest.approx(0.0, abs=ABS)


def test_two_and_four_anchors_agree_on_fused_pose() -> None:
    body = pose(x=1.55, y=1.1, yaw=0.7)
    anchors = kitchen_anchors()
    two = _solve(body, _observe(body, anchors, 2))
    four = _solve(body, _observe(body, anchors, 4))
    assert two.pose.estimate is not None and four.pose.estimate is not None
    _approx_pose(two.pose.estimate, body)
    _approx_pose(four.pose.estimate, body)
    assert two.translation_rmse_m == pytest.approx(0.0, abs=1e-8)
    assert four.translation_rmse_m == pytest.approx(0.0, abs=1e-8)


def test_anchor_disagreement_is_reported() -> None:
    body = pose(x=1.0, y=1.0)
    observations = list(_observe(body, kitchen_anchors(), 2))
    noisy = observations[1]
    observations[1] = FiducialObservation(
        anchor_id=noisy.anchor_id,
        pose=pose(
            x=noisy.pose.position.x + 0.25,
            y=noisy.pose.position.y,
            z=noisy.pose.position.z,
            yaw=noisy.pose.orientation.yaw,
        ),
        timestamp=noisy.timestamp,
        confidence=noisy.confidence,
    )
    solution = _solve(body, observations)
    assert solution.translation_rmse_m is not None
    assert solution.translation_rmse_m > LOCALIZED_TRANSLATION_M
    assert solution.pose.quality == "degraded"
    assert solution.pose.uncertainty is not None
    assert solution.pose.uncertainty.x_sigma is not None
    assert solution.pose.uncertainty.x_sigma > 0


def test_no_observations_is_lost() -> None:
    solution = _solve(pose(x=1.0, y=1.0), ())
    assert solution.pose.quality == "lost"
    assert solution.pose.estimate is None


def test_unknown_anchor_observation_is_ignored() -> None:
    body = pose(x=1.2, y=0.9, yaw=0.2)
    known = _observe(body, kitchen_anchors(), 1)
    ghost = FiducialObservation(
        anchor_id="unsurveyed",
        pose=pose(x=0.3),
        timestamp=STAMP,
    )
    solution = _solve(body, (ghost,) + known)
    assert solution.pose.quality == "localized"
    assert solution.pose.estimate is not None
    _approx_pose(solution.pose.estimate, body)


def test_kitchen_anchor_fixture_is_not_origin_or_limits() -> None:
    anchors = kitchen_anchors()
    xs = {round(item.position.x, 4) for item in anchors}
    ys = {round(item.position.y, 4) for item in anchors}
    assert as_meters({"value": 137, "unit": "cm"}) in xs
    assert as_meters({"value": 105, "unit": "cm"}) in ys
    assert 0.0 in xs and 0.0 in ys
    assert (0.0, 0.0) not in {
        (round(item.position.x, 4), round(item.position.y, 4)) for item in anchors
    }


def test_accuracy_within_localized_budget_for_arbitrary_yaw() -> None:
    anchors = kitchen_anchors()
    errors = []
    for yaw in (0.0, 0.4, RIGHT, math.pi, -1.2):
        body = pose(x=2.05, y=1.33, yaw=yaw)
        solution = _solve(body, _observe(body, anchors, 4))
        assert solution.pose.estimate is not None
        dx = solution.pose.estimate.position.x - body.position.x
        dy = solution.pose.estimate.position.y - body.position.y
        translation = math.hypot(dx, dy)
        yaw_error = abs((solution.pose.estimate.orientation.yaw - yaw + math.pi) % (2 * math.pi) - math.pi)
        errors.append((translation, yaw_error))
        assert translation < LOCALIZED_TRANSLATION_M
        assert yaw_error < LOCALIZED_YAW_RAD
        assert solution.pose.quality == "localized"
    assert errors


def test_localize_in_space_reads_embedded_anchors() -> None:
    body = pose(x=1.4, y=0.9, yaw=0.3)
    space = kitchen_space()
    observations = _observe(body, anchors_from_space(space), 4)
    solution = localize_in_space(
        agent=AGENT,
        space_record=space,
        observations=observations,
        camera_in_body=CAMERA,
        timestamp=STAMP,
    )
    assert solution.pose.quality == "localized"
    assert solution.pose.estimate is not None
    _approx_pose(solution.pose.estimate, body)
