"""Pure surveyed-anchor localization. No detector, ROS, or graph traversal."""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .anchors import SurveyedAnchor, anchors_from_space, global_anchor_id
from .contracts import SpatialContractError
from .observation import FiducialObservation
from .pose import PoseUncertainty, RuntimePose
from .transforms import Pose, compose_pose, invert_pose, pose

SOURCE = "surveyed_fiducials"
LOCALIZED_TRANSLATION_M = 0.05
LOCALIZED_YAW_RAD = math.radians(5.0)


@dataclass(frozen=True)
class LocalizationSolution:
    """Body pose in the named space, plus residual statistics for the harness."""

    pose: RuntimePose
    samples: tuple[Pose, ...]
    translation_rmse_m: float | None = None
    yaw_rmse_rad: float | None = None


def localize_from_fiducials(
    *,
    agent: str,
    space: str,
    anchors: Sequence[SurveyedAnchor],
    observations: Sequence[FiducialObservation],
    camera_in_body: Pose,
    timestamp: str | None = None,
) -> LocalizationSolution:
    """Solve MicroDuck body pose in ``space`` from surveyed + observed anchors.

    Each observation is the anchor pose in the camera frame. ``camera_in_body``
    is the camera pose in the robot body. The duck's start pose is not an input.
    """
    catalog = _anchor_catalog(anchors, space)
    samples: list[Pose] = []
    used: list[FiducialObservation] = []
    for observation in observations:
        known = catalog.get(observation.anchor_id)
        if known is None:
            continue
        samples.append(_body_in_space(known, observation.pose, camera_in_body))
        used.append(observation)
    if not samples:
        return LocalizationSolution(
            pose=RuntimePose(agent=agent, space=space, quality="lost"),
            samples=(),
        )
    fused = _fuse_poses(samples, used)
    translation_rmse, yaw_rmse = _residuals(fused, samples)
    quality = _quality(translation_rmse, yaw_rmse)
    stamp = timestamp or max(item.timestamp for item in used)
    survey_floor = _survey_floor(used, catalog)
    return LocalizationSolution(
        pose=RuntimePose(
            agent=agent,
            space=space,
            quality=quality,
            estimate=fused,
            timestamp=stamp,
            source=SOURCE,
            uncertainty=_uncertainty(samples, survey_floor),
        ),
        samples=tuple(samples),
        translation_rmse_m=translation_rmse,
        yaw_rmse_rad=yaw_rmse,
    )


def localize_in_space(
    *,
    agent: str,
    space_record: Mapping[str, Any],
    observations: Sequence[FiducialObservation],
    camera_in_body: Pose,
    timestamp: str | None = None,
) -> LocalizationSolution:
    """Run the solver using anchors embedded on ``space.coordinate``."""
    space_id = space_record.get("id")
    if not isinstance(space_id, str):
        raise SpatialContractError("space record requires id")
    return localize_from_fiducials(
        agent=agent,
        space=space_id,
        anchors=anchors_from_space(space_record),
        observations=observations,
        camera_in_body=camera_in_body,
        timestamp=timestamp,
    )


def synthetic_observation(
    *,
    body_in_space: Pose,
    camera_in_body: Pose,
    anchor: SurveyedAnchor,
    timestamp: str,
    confidence: float | None = 1.0,
) -> FiducialObservation:
    """Noise-free observation of an anchor from a known body pose. Not a detector."""
    camera_in_space = compose_pose(body_in_space, camera_in_body)
    anchor_in_camera = compose_pose(invert_pose(camera_in_space), _surveyed_pose(anchor))
    return FiducialObservation(
        anchor_id=anchor.id,
        pose=anchor_in_camera,
        timestamp=timestamp,
        confidence=confidence,
    )


def _body_in_space(anchor: SurveyedAnchor, anchor_in_camera: Pose, camera_in_body: Pose) -> Pose:
    camera_in_space = compose_pose(_surveyed_pose(anchor), invert_pose(anchor_in_camera))
    return compose_pose(camera_in_space, invert_pose(camera_in_body))


def _surveyed_pose(anchor: SurveyedAnchor) -> Pose:
    if anchor.orientation is None:
        raise SpatialContractError(
            f"localization requires orientation on {anchor.id}"
        )
    return Pose(anchor.position, anchor.orientation)


def _anchor_catalog(anchors: Sequence[SurveyedAnchor], space: str) -> dict[str, SurveyedAnchor]:
    catalog: dict[str, SurveyedAnchor] = {}
    for anchor in anchors:
        if anchor.space != space:
            continue
        catalog[anchor.id] = anchor
        catalog[global_anchor_id(anchor.space, anchor.id)] = anchor
    return catalog


def _fuse_poses(
    samples: Sequence[Pose],
    observations: Sequence[FiducialObservation],
) -> Pose:
    weights = [
        1.0 if observation.confidence is None else float(observation.confidence)
        for observation in observations
    ]
    total = sum(weights)
    if total <= 0:
        weights = [1.0] * len(samples)
        total = float(len(samples))
    x = sum(w * item.position.x for w, item in zip(weights, samples)) / total
    y = sum(w * item.position.y for w, item in zip(weights, samples)) / total
    z = sum(w * item.position.z for w, item in zip(weights, samples)) / total
    yaw = _circular_mean([item.orientation.yaw for item in samples], weights)
    pitch = _circular_mean([item.orientation.pitch for item in samples], weights)
    roll = _circular_mean([item.orientation.roll for item in samples], weights)
    return pose(x=x, y=y, z=z, yaw=yaw, pitch=pitch, roll=roll)


def _residuals(fused: Pose, samples: Sequence[Pose]) -> tuple[float, float]:
    translation = [
        math.dist(fused.position.as_tuple(), item.position.as_tuple())
        for item in samples
    ]
    yaw = [_wrap(item.orientation.yaw - fused.orientation.yaw) for item in samples]
    return (
        math.sqrt(sum(value * value for value in translation) / len(translation)),
        math.sqrt(sum(value * value for value in yaw) / len(yaw)),
    )


def _quality(translation_rmse: float, yaw_rmse: float) -> str:
    if translation_rmse <= LOCALIZED_TRANSLATION_M and yaw_rmse <= LOCALIZED_YAW_RAD:
        return "localized"
    return "degraded"


def _uncertainty(samples: Sequence[Pose], survey_floor: float | None) -> PoseUncertainty | None:
    if len(samples) == 1:
        if survey_floor is None:
            return None
        return PoseUncertainty(x_sigma=survey_floor, y_sigma=survey_floor, z_sigma=survey_floor)
    xs = [item.position.x for item in samples]
    ys = [item.position.y for item in samples]
    zs = [item.position.z for item in samples]
    yaws = [item.orientation.yaw for item in samples]
    return PoseUncertainty(
        x_sigma=_sample_std(xs),
        y_sigma=_sample_std(ys),
        z_sigma=_sample_std(zs),
        yaw_sigma=_circular_std(yaws),
    )


def _survey_floor(
    observations: Sequence[FiducialObservation],
    catalog: Mapping[str, SurveyedAnchor],
) -> float | None:
    floors = [
        catalog[observation.anchor_id].survey.uncertainty_m
        for observation in observations
        if observation.anchor_id in catalog
        and catalog[observation.anchor_id].survey is not None
        and catalog[observation.anchor_id].survey.uncertainty_m is not None
    ]
    return min(floors) if floors else None


def _circular_mean(angles: Sequence[float], weights: Sequence[float]) -> float:
    total = sum(weights)
    sine = sum(weight * math.sin(angle) for weight, angle in zip(weights, angles)) / total
    cosine = sum(weight * math.cos(angle) for weight, angle in zip(weights, angles)) / total
    return math.atan2(sine, cosine)


def _circular_std(angles: Sequence[float]) -> float:
    mean = _circular_mean(angles, [1.0] * len(angles))
    return _sample_std([_wrap(angle - mean) for angle in angles])


def _sample_std(values: Sequence[float]) -> float:
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance)


def _wrap(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi
