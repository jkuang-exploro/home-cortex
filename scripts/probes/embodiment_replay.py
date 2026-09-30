"""Deterministic fused-telemetry simulator and Epic 1 JSON inspector.

This engineering harness knows ground truth so that reported physical p95 can
be checked. It does not simulate sensors, localization, or robot actuators.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from home_cortex.agents.embodiments import EmbodimentCatalog, EmbodimentConnections
from home_cortex.agents.session import EmbodimentSession
from home_cortex.spatial.embodiment import embodiment_from_mapping
from home_cortex.spatial.occupancy import nominal_box_envelope, translation_p95_envelope
from home_cortex.spatial.presence import EmbodimentPresence
from home_cortex.spatial.telemetry import (
    PhysicalTelemetry, PhysicalValue, RealtimeTransform, box_corners_in_space,
    embodiment_point_in_space, physical_telemetry_as_mapping,
    physical_telemetry_from_mapping,
)
from home_cortex.spatial.transforms import (
    Pose, Position, pose, pose_as_mapping, pose_from_mapping, transform_pose,
)
from scripts.maintenance.embodiment_client import SimulatedEmbodimentClient


SCENARIOS = (
    "stationary", "straight", "square", "circle", "rotation",
    "forward_rotate", "random",
)
START = datetime(2026, 9, 28, 18, 0, tzinfo=timezone.utc)
BODY_ID = "embodiment:microduck-01"
SPACE_ID = "space:kitchen"
CHILD_IN_BODY = pose(x=0.18, z=0.12)
_FIELDS = ("x", "y", "z", "yaw", "pitch", "roll")


@dataclass(frozen=True)
class ErrorProfile:
    position_sigma_m: float = 0.0
    orientation_sigma_rad: float = 0.0
    position_bias_m: float = 0.0
    orientation_bias_rad: float = 0.0
    p95_scale: float = 1.0
    calibrated: bool = True

    def __post_init__(self) -> None:
        values = (self.position_sigma_m, self.orientation_sigma_rad,
                  self.position_bias_m, self.orientation_bias_rad, self.p95_scale)
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("error profile values must be finite and non-negative")
        if self.calibrated and self.p95_scale < 1:
            raise ValueError("a calibrated fixture cannot underreport its p95 scale")


PROFILES = {
    "exact": ErrorProfile(),
    "low": ErrorProfile(position_sigma_m=0.006, orientation_sigma_rad=0.006),
    "moderate": ErrorProfile(position_sigma_m=0.03, orientation_sigma_rad=0.025),
    "large": ErrorProfile(position_sigma_m=0.15, orientation_sigma_rad=0.12),
    "biased": ErrorProfile(position_sigma_m=0.01, orientation_sigma_rad=0.01,
                           position_bias_m=0.25, orientation_bias_rad=0.2),
    "underreported": ErrorProfile(position_sigma_m=0.01, orientation_sigma_rad=0.01,
                                  position_bias_m=0.25, orientation_bias_rad=0.2,
                                  p95_scale=0.1, calibrated=False),
}


def synthetic_body() -> dict[str, Any]:
    return {
        "id": BODY_ID,
        "name": "MicroDuck",
        "agent_id": "agent:butler",
        "capabilities": ["mobility.move", "vision.observe", "audio.speak"],
        "local_frame": {"forward": "+x", "left": "+y", "up": "+z"},
        "geometry": {"box": {
            "length_m": 0.32, "width_m": 0.24, "height_m": 0.18,
            "center": {"x": 0.04, "y": 0.0, "z": 0.09},
        }},
    }


def ground_truth_path(scenario: str, *, count: int, seed: int) -> tuple[Pose, ...]:
    """Exact body poses within one synthetic kitchen; one-second sample period."""
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario {scenario!r}")
    if count < 2:
        raise ValueError("count must be at least 2")
    result: list[Pose] = []
    rng = random.Random(seed)
    x, y, yaw = 2.0, 2.0, 0.0
    for index in range(count):
        progress = index / (count - 1)
        if scenario == "stationary":
            current = pose(x=2, y=2)
        elif scenario == "straight":
            current = pose(x=1.0 + 2.0 * progress, y=2.0)
        elif scenario == "square":
            leg = min(3, int(4 * progress))
            part = 4 * progress - leg
            segments = (
                (1 + part, 1, 0), (2, 1 + part, math.pi / 2),
                (2 - part, 2, math.pi), (1, 2 - part, -math.pi / 2),
            )
            sx, sy, heading = segments[leg]
            current = pose(x=sx, y=sy, yaw=heading)
        elif scenario == "circle":
            angle = 2 * math.pi * progress
            current = pose(x=2 + 0.6 * math.cos(angle),
                           y=2 + 0.6 * math.sin(angle), yaw=angle + math.pi / 2)
        elif scenario == "rotation":
            current = pose(x=2, y=2, yaw=2 * math.pi * progress)
        elif scenario == "forward_rotate":
            if index:
                yaw += 1.2 / (count - 1)
                x += (1.5 / (count - 1)) * math.cos(yaw)
                y += (1.5 / (count - 1)) * math.sin(yaw)
            current = pose(x=x, y=y, yaw=yaw)
        else:
            if index:
                yaw += rng.uniform(-0.25, 0.25)
                x = min(3.2, max(0.8, x + 0.08 * math.cos(yaw)))
                y = min(3.2, max(0.8, y + 0.08 * math.sin(yaw)))
            current = pose(x=x, y=y, yaw=yaw)
        result.append(current)
    return tuple(result)


def fused_sample(
    truth: Pose, *, measured_at: datetime, profile: ErrorProfile, rng: random.Random,
) -> PhysicalTelemetry:
    """Already-fused estimate; bias grows the reported p95 in calibrated fixtures."""
    values = (
        truth.position.x, truth.position.y, truth.position.z,
        truth.orientation.yaw, truth.orientation.pitch, truth.orientation.roll,
    )
    estimates = []
    for index, actual in enumerate(values):
        translation = index < 3
        sigma = profile.position_sigma_m if translation else profile.orientation_sigma_rad
        bias = profile.position_bias_m if translation else profile.orientation_bias_rad
        error = bias + (rng.gauss(0, sigma) if sigma else 0.0)
        p95 = profile.p95_scale * (bias + 1.96 * sigma)
        estimates.append(PhysicalValue(actual + error, p95))
    return PhysicalTelemetry(BODY_ID, SPACE_ID, measured_at, "valid", RealtimeTransform(*estimates))


def build_tape(
    scenario: str, *, count: int = 400, seed: int = 7, profile: str = "low",
) -> dict[str, Any]:
    """Serializable ground truth and canonical telemetry for repeatable replay."""
    if profile not in PROFILES:
        raise ValueError(f"unknown error profile {profile!r}")
    rng = random.Random(seed + 10_000)
    samples = []
    for index, truth in enumerate(ground_truth_path(scenario, count=count, seed=seed)):
        measured_at = START + timedelta(seconds=index)
        sample = fused_sample(truth, measured_at=measured_at, profile=PROFILES[profile], rng=rng)
        samples.append({"ground_truth": pose_as_mapping(truth),
                        "telemetry": physical_telemetry_as_mapping(sample)})
    return {"scenario": scenario, "profile": profile, "seed": seed,
            "space_id": SPACE_ID, "space_axes": {
                "x": [1, 0, 0], "y": [0, 1, 0], "z": [0, 0, 1],
            }, "anchors": [], "embodiment": synthetic_body(), "samples": samples}


def _angular_error(estimated: float, actual: float) -> float:
    return math.atan2(math.sin(estimated - actual), math.cos(estimated - actual))


def _point_mapping(point: Position) -> dict[str, float]:
    return dict(zip(("x", "y", "z"), point.as_tuple()))


def _status_mapping(session: EmbodimentSession, *, now: datetime) -> dict[str, Any]:
    state = session.presence.latest(BODY_ID, now=now)
    return {"online": session.connections.is_connected(BODY_ID),
            "valid": state.status.valid, "fresh": state.status.fresh,
            "stale": state.status.stale, "unavailable": state.status.unavailable}


def replay_tape(tape: dict[str, Any]) -> dict[str, Any]:
    """Drive the actual Home Cortex session/presence contracts from a tape."""
    body = embodiment_from_mapping(tape["embodiment"])
    if body.id != BODY_ID or tape["space_id"] != SPACE_ID:
        raise ValueError("this V1 replay supports only its single declared body and space")
    axes = {"x": [1, 0, 0], "y": [0, 1, 0], "z": [0, 0, 1]}
    if tape.get("space_axes", axes) != axes:
        raise ValueError("this V1 replay uses the kitchen's identity space basis")
    catalog = EmbodimentCatalog([body])
    presence = EmbodimentPresence(
        embodiment_ids=catalog.embodiment_ids, space_ids=[SPACE_ID], stale_after_s=2,
    )
    session = EmbodimentSession(catalog, EmbodimentConnections(catalog), presence)
    client = SimulatedEmbodimentClient(session, BODY_ID)
    opened = client.register(["mobility.move", "vision.observe"], now=START)
    if opened["agent_id"] != "agent:butler":
        raise ValueError("the scenario must resolve to the existing butler agent")
    errors: list[dict[str, float]] = []
    coverage: list[dict[str, bool]] = []
    recent: list[dict[str, float]] = []
    inspection: dict[str, Any] = {}
    last_sample: PhysicalTelemetry | None = None
    last_truth: Pose | None = None
    for entry in tape["samples"]:
        truth = pose_from_mapping(entry["ground_truth"])
        sample = physical_telemetry_from_mapping(entry["telemetry"])
        if sample.embodiment_id != body.id or sample.space_id != SPACE_ID or sample.transform is None:
            raise ValueError("replay entries require a valid pose in the declared space")
        observed = sample.measured_at + timedelta(milliseconds=100)
        admission = client.send_telemetry(physical_telemetry_as_mapping(sample), now=observed)
        if admission["telemetry"]["disposition"] != "accepted":
            raise ValueError("replay samples must have increasing measurement times")
        current = presence.latest(BODY_ID, now=observed)
        if current.telemetry != sample or not current.status.fresh:
            raise AssertionError("ingested telemetry did not become the fresh current sample")
        transform = sample.transform
        estimate_values = tuple(getattr(transform, field).value for field in _FIELDS)
        truth_values = (*truth.position.as_tuple(), truth.orientation.yaw,
                        truth.orientation.pitch, truth.orientation.roll)
        residuals = tuple(
            estimated - actual if index < 3 else _angular_error(estimated, actual)
            for index, (estimated, actual) in enumerate(zip(estimate_values, truth_values))
        )
        errors.append(dict(zip(_FIELDS, residuals)))
        coverage.append({field: abs(error) <= getattr(transform, field).p95 + 1e-12
                         for field, error in zip(_FIELDS, residuals)})
        corners = box_corners_in_space(body.geometry, body.local_frame, transform)
        nominal = nominal_box_envelope(body.geometry, body.local_frame, transform)
        inflated = translation_p95_envelope(nominal, transform)
        camera = embodiment_point_in_space(CHILD_IN_BODY.position, transform)
        camera_pose = transform_pose(CHILD_IN_BODY, transform.nominal_pose())
        if any(abs(a - b) > 1e-9 for a, b in zip(camera.as_tuple(), camera_pose.position.as_tuple())):
            raise AssertionError("child-frame transform disagrees with the nominal pose")
        recent.append(_point_mapping(transform.nominal_pose().position))
        inspection = {
            "ground_truth": entry["ground_truth"],
            "telemetry": entry["telemetry"],
            "box_corners_in_space": [_point_mapping(point) for point in corners],
            "nominal_envelope": nominal.as_mapping(),
            "translation_p95_envelope": inflated.as_mapping(),
            "camera_in_body": pose_as_mapping(CHILD_IN_BODY),
            "camera_in_space": pose_as_mapping(camera_pose),
            "local_origin_in_space": _point_mapping(
                embodiment_point_in_space(Position(), transform)
            ),
        }
        last_sample, last_truth = sample, truth
    if last_sample is None or last_truth is None:
        raise ValueError("replay tape requires samples")
    stale_at = last_sample.measured_at + timedelta(seconds=4)
    transitions = [{"event": "latest", **_status_mapping(session, now=last_sample.measured_at)},
                   {"event": "stale", **_status_mapping(session, now=stale_at)}]
    missing_at = last_sample.measured_at + timedelta(seconds=5)
    client.send_no_estimate(measured_at=missing_at.isoformat(), space_id=SPACE_ID, now=missing_at)
    transitions.append({"event": "no_estimate", **_status_mapping(session, now=missing_at)})
    recovered_at = missing_at + timedelta(seconds=1)
    recovered = fused_sample(last_truth, measured_at=recovered_at,
                             profile=PROFILES["exact"], rng=random.Random(0))
    client.send_telemetry(physical_telemetry_as_mapping(recovered), now=recovered_at)
    transitions.append({"event": "recovery", **_status_mapping(session, now=recovered_at)})
    client.disconnect(now=recovered_at + timedelta(seconds=1))
    transitions.append({"event": "disconnect", **_status_mapping(session, now=recovered_at + timedelta(seconds=1))})
    client.reconnect(["vision.observe"], now=recovered_at + timedelta(seconds=2))
    transitions.append({"event": "reconnect", **_status_mapping(session, now=recovered_at + timedelta(seconds=2))})
    if catalog.get(BODY_ID).agent_id != "agent:butler" or catalog.embodiment_ids != {BODY_ID}:
        raise AssertionError("runtime lifecycle changed persistent embodiment identity")
    count = len(errors)
    position_errors = [math.sqrt(sum(row[axis] ** 2 for axis in ("x", "y", "z")))
                       for row in errors]
    orientation_errors = [math.sqrt(sum(row[axis] ** 2 for axis in ("yaw", "pitch", "roll")))
                          for row in errors]
    p95 = {field: max(entry["telemetry"]["transform"][field]["p95"]
                      for entry in tape["samples"]) for field in _FIELDS}
    coverage_by_axis = {field: sum(row[field] for row in coverage) / count for field in _FIELDS}
    return {
        "scenario": tape["scenario"], "profile": tape["profile"], "seed": tape["seed"],
        "space_id": SPACE_ID, "embodiment_id": BODY_ID, "agent_id": body.agent_id,
        "sample_count": count,
        "metrics": {
            "position_error_mean_m": sum(position_errors) / count,
            "position_error_max_m": max(position_errors),
            "orientation_error_mean_rad": sum(orientation_errors) / count,
            "orientation_error_max_rad": max(orientation_errors),
            "reported_p95": p95,
            "coverage_by_axis": coverage_by_axis,
            "coverage_rate": sum(sum(row.values()) for row in coverage) / (6 * count),
            "transform_correct": True,
        },
        "transitions": transitions,
        "inspection": {**inspection, "space_axes": axes,
                       "anchors": tape.get("anchors", []), "recent_trajectory": recent[-12:]},
        "persistent_identity_after_reconnect": catalog.get(BODY_ID).id,
        "persistent_agent_after_reconnect": catalog.get(BODY_ID).agent_id,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=SCENARIOS, default="square")
    parser.add_argument("--profile", choices=tuple(PROFILES), default="low")
    parser.add_argument("--count", type=int, default=400)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--record", type=Path, help="Write a ground-truth and telemetry replay tape")
    parser.add_argument("--replay", type=Path, help="Read a previously recorded replay tape")
    parser.add_argument("--output", type=Path, help="Write the JSON inspection report")
    args = parser.parse_args(argv)
    tape = json.loads(args.replay.read_text(encoding="utf-8")) if args.replay else build_tape(
        args.scenario, count=args.count, seed=args.seed, profile=args.profile,
    )
    if args.record:
        args.record.write_text(json.dumps(tape, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = replay_tape(tape)
    encoded = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
    else:
        print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
