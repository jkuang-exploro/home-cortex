"""Pre-hardware Epic 1 replay against canonical session and spatial contracts."""
import json
import math
from pathlib import Path

import pytest

from home_cortex.spatial.embodiment import embodiment_from_mapping
from home_cortex.spatial.occupancy import (
    envelope_of_points, nominal_box_envelope, translation_p95_envelope,
)
from home_cortex.spatial.telemetry import (
    PhysicalValue, RealtimeTransform, box_corners_in_space,
    embodiment_point_in_space,
)
from home_cortex.spatial.transforms import pose, transform_pose
from scripts.probes.embodiment_replay import (
    BODY_ID, PROFILES, SCENARIOS, ErrorProfile, build_tape, ground_truth_path,
    main, replay_tape, synthetic_body,
)


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_every_trajectory_is_deterministic_and_single_space(scenario: str) -> None:
    first = build_tape(scenario, count=24, seed=17, profile="low")
    assert first == build_tape(scenario, count=24, seed=17, profile="low")
    assert all(entry["telemetry"]["space_id"] == "space:kitchen"
               for entry in first["samples"])
    assert all(0.5 <= truth.position.x <= 3.5 and 0.5 <= truth.position.y <= 3.5
               for truth in ground_truth_path(scenario, count=24, seed=17))
    report = replay_tape(first)
    assert report["sample_count"] == 24
    assert report["metrics"]["transform_correct"] is True
    assert report["persistent_identity_after_reconnect"] == BODY_ID
    assert report["persistent_agent_after_reconnect"] == "agent:butler"


def test_exact_replay_exercises_full_lifecycle_and_inspection() -> None:
    report = replay_tape(build_tape("rotation", count=16, seed=7, profile="exact"))
    metrics = report["metrics"]
    assert metrics["position_error_max_m"] == 0
    assert metrics["orientation_error_max_rad"] < 1e-15
    assert metrics["coverage_rate"] == 1
    assert all(value == 0 for value in metrics["reported_p95"].values())
    transitions = report["transitions"]
    assert [item["event"] for item in transitions] == [
        "latest", "stale", "no_estimate", "recovery", "disconnect", "reconnect",
    ]
    assert transitions[0]["fresh"]
    assert transitions[1]["stale"]
    assert transitions[2]["unavailable"] and transitions[2]["online"]
    assert transitions[3]["fresh"]
    assert not transitions[4]["online"]
    assert transitions[5]["online"]
    inspection = report["inspection"]
    assert len(inspection["box_corners_in_space"]) == 8
    assert len(inspection["recent_trajectory"]) == 12
    assert inspection["local_origin_in_space"] == {
        "x": inspection["telemetry"]["transform"]["x"]["value"],
        "y": inspection["telemetry"]["transform"]["y"]["value"],
        "z": inspection["telemetry"]["transform"]["z"]["value"],
    }


def test_large_injected_error_has_large_reported_uncertainty() -> None:
    low = replay_tape(build_tape("straight", count=400, seed=11, profile="low"))["metrics"]
    moderate = replay_tape(build_tape("straight", count=400, seed=11, profile="moderate"))["metrics"]
    large = replay_tape(build_tape("straight", count=400, seed=11, profile="large"))["metrics"]
    biased = replay_tape(build_tape("straight", count=400, seed=11, profile="biased"))["metrics"]
    underreported = replay_tape(build_tape("straight", count=400, seed=11, profile="underreported"))["metrics"]
    assert low["position_error_mean_m"] < moderate["position_error_mean_m"] < large["position_error_mean_m"]
    assert low["reported_p95"]["x"] < moderate["reported_p95"]["x"] < large["reported_p95"]["x"]
    assert biased["reported_p95"]["x"] >= 0.25
    assert biased["reported_p95"]["yaw"] >= 0.2
    assert biased["coverage_rate"] >= 0.95
    assert underreported["reported_p95"]["x"] < biased["reported_p95"]["x"]
    assert underreported["coverage_rate"] < 0.25


def test_calibrated_profile_cannot_silently_underreport() -> None:
    with pytest.raises(ValueError, match="cannot underreport"):
        ErrorProfile(position_sigma_m=0.1, p95_scale=0.1, calibrated=True)
    assert PROFILES["underreported"].calibrated is False


def test_box_envelope_and_child_pose_keep_geometry_separate_from_uncertainty() -> None:
    body = embodiment_from_mapping(synthetic_body())
    transform = RealtimeTransform(
        PhysicalValue(1, 0.05), PhysicalValue(2, 0.07), PhysicalValue(0, 0.02),
        PhysicalValue(math.pi / 2, 0.1), PhysicalValue(0, 0.01), PhysicalValue(0, 0.01),
    )
    corners = box_corners_in_space(body.geometry, body.local_frame, transform)
    nominal = nominal_box_envelope(body.geometry, body.local_frame, transform)
    assert nominal == envelope_of_points(corners)
    assert nominal.minimum.x == pytest.approx(0.88)
    assert nominal.maximum.x == pytest.approx(1.12)
    assert nominal.minimum.y == pytest.approx(1.88)
    assert nominal.maximum.y == pytest.approx(2.2)
    inflated = translation_p95_envelope(nominal, transform)
    assert inflated.minimum.x == pytest.approx(0.83)
    assert inflated.maximum.y == pytest.approx(2.27)
    assert body.geometry.length_m == 0.32
    child = pose(x=0.18, z=0.12)
    child_in_space = transform_pose(child, transform.nominal_pose())
    point_in_space = embodiment_point_in_space(child.position, transform)
    assert child_in_space.position == pytest.approx(point_in_space)
    assert child_in_space.position.x == pytest.approx(1)
    assert child_in_space.position.y == pytest.approx(2.18)


def test_recorded_tape_replays_to_identical_json_report(tmp_path: Path) -> None:
    tape_path = tmp_path / "tape.json"
    report_path = tmp_path / "report.json"
    replay_path = tmp_path / "replay.json"
    assert main(["--scenario", "square", "--profile", "low", "--count", "12",
                 "--seed", "23", "--record", str(tape_path), "--output", str(report_path)]) == 0
    assert main(["--replay", str(tape_path), "--output", str(replay_path)]) == 0
    assert json.loads(report_path.read_text()) == json.loads(replay_path.read_text())
    assert json.loads(tape_path.read_text())["embodiment"]["agent_id"] == "agent:butler"


def test_inspector_preserves_optional_space_anchors() -> None:
    tape = build_tape("stationary", count=2, seed=7, profile="exact")
    tape["anchors"] = [{"id": "anchor:kitchen-a", "position": {"x": 1, "y": 1, "z": 0}}]
    report = replay_tape(tape)
    assert report["inspection"]["anchors"] == tape["anchors"]
