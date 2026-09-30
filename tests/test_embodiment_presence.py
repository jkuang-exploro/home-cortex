"""Latest fused-telemetry sample, admission, and derived freshness."""
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from home_cortex.api.app import app
from home_cortex.spatial.presence import (
    EmbodimentPresence, TelemetryAdmissionError, admission_as_mapping,
    derive_telemetry_status, embodiment_ids_from_node_file, record_ids_from_node_file,
    telemetry_state_as_mapping,
)
from home_cortex.spatial.primitives import SpatialContractError
from home_cortex.spatial.telemetry import physical_telemetry_from_mapping


NOW = datetime(2026, 9, 27, 19, 0, 1, tzinfo=timezone.utc)
MEASURED = "2026-09-27T19:00:00+00:00"
_BASE = datetime(2026, 9, 27, 19, 0, tzinfo=timezone.utc)


def _sample(*, offset_s: float = 0, validity: str = "valid", space: str | None = "space:kitchen",
            x: float = 1.372) -> dict:
    measured = _BASE + timedelta(seconds=offset_s)
    return {
        "embodiment_id": "embodiment:microduck-01",
        "space_id": space,
        "measured_at": measured.isoformat(),
        "validity": validity,
        "transform": None if validity != "valid" else {
            name: {"value": value, "p95": uncertainty}
            for name, value, uncertainty in (
                ("x", x, 0.024), ("y", 0.814, 0.021), ("z", 0.09, 0.01),
                ("yaw", 1.118, 0.037), ("pitch", 0.0, 0.01), ("roll", 0.0, 0.01),
            )
        },
    }


def _presence() -> EmbodimentPresence:
    return EmbodimentPresence(
        embodiment_ids=["embodiment:microduck-01"],
        space_ids=["space:kitchen"],
        stale_after_s=2,
    )


def test_new_sample_is_current_and_serializes() -> None:
    presence = _presence()
    admission = presence.submit(_sample(), now=NOW)
    assert admission.disposition == "accepted"
    assert admission.state.status.valid and admission.state.status.fresh
    assert admission.state.telemetry is not None
    assert admission.state.telemetry.transform is not None
    assert admission.state.telemetry.transform.x.value == 1.372
    payload = json.loads(json.dumps(admission_as_mapping(admission)))
    assert payload["disposition"] == "accepted"
    restored = physical_telemetry_from_mapping(payload["state"]["telemetry"])
    assert restored == admission.state.telemetry
    assert telemetry_state_as_mapping(presence.latest("embodiment:microduck-01", now=NOW)) == payload["state"]


def test_older_and_out_of_order_samples_do_not_replace_newer_state() -> None:
    presence = _presence()
    assert presence.submit(_sample(offset_s=0), now=NOW).disposition == "accepted"
    assert presence.submit(_sample(offset_s=2, x=3), now=NOW).disposition == "accepted"
    ignored = presence.submit(_sample(offset_s=1, x=9), now=NOW)
    assert ignored.disposition == "ignored"
    assert ignored.state.telemetry is not None
    assert ignored.state.telemetry.measured_at == datetime(2026, 9, 27, 19, 0, 2, tzinfo=timezone.utc)
    assert ignored.state.telemetry.transform is not None
    assert ignored.state.telemetry.transform.x.value == 3


def test_duplicate_sample_is_deterministic() -> None:
    presence = _presence()
    first = presence.submit(_sample(), now=NOW)
    second = presence.submit(_sample(), now=NOW)
    assert second.disposition == "duplicate"
    assert second.state.telemetry == first.state.telemetry
    assert presence.latest("embodiment:microduck-01", now=NOW).telemetry == first.state.telemetry


def test_same_timestamp_with_different_content_is_rejected() -> None:
    presence = _presence()
    presence.submit(_sample(), now=NOW)
    with pytest.raises(TelemetryAdmissionError, match="already current") as error:
        presence.submit(_sample(x=4), now=NOW)
    assert error.value.code == "conflicting_sample"
    current = presence.latest("embodiment:microduck-01", now=NOW)
    assert current.telemetry is not None and current.telemetry.transform is not None
    assert current.telemetry.transform.x.value == 1.372


@pytest.mark.parametrize("bad", [-0.01, math.nan, math.inf, True])
def test_invalid_uncertainty_does_not_change_state(bad) -> None:
    presence = _presence()
    presence.submit(_sample(), now=NOW)
    payload = _sample(offset_s=5)
    payload["transform"]["x"]["p95"] = bad
    with pytest.raises(SpatialContractError):
        presence.submit(payload, now=NOW)
    current = presence.latest("embodiment:microduck-01", now=NOW)
    assert current.telemetry is not None
    assert current.telemetry.measured_at.second == 0


def test_malformed_orientation_and_timestamp_are_rejected() -> None:
    presence = _presence()
    missing_roll = _sample()
    del missing_roll["transform"]["roll"]
    with pytest.raises(SpatialContractError, match="transform"):
        presence.submit(missing_roll, now=NOW)
    bad_time = _sample()
    bad_time["measured_at"] = "2026-09-27T19:00:00"
    with pytest.raises(SpatialContractError, match="measured_at"):
        presence.submit(bad_time, now=NOW)
    assert presence.latest("embodiment:microduck-01", now=NOW).telemetry is None


def test_missing_estimate_replaces_only_when_newer() -> None:
    presence = _presence()
    empty = presence.latest("embodiment:microduck-01", now=NOW)
    assert empty.telemetry is None and empty.status.unavailable and not empty.status.valid
    presence.submit(_sample(offset_s=1), now=NOW)
    lost = presence.submit(_sample(offset_s=2, validity="no_estimate", space=None), now=NOW)
    assert lost.disposition == "accepted"
    assert lost.state.telemetry is not None and lost.state.telemetry.transform is None
    assert lost.state.status.unavailable and not lost.state.status.fresh
    ignored = presence.submit(_sample(offset_s=0, validity="no_estimate", space=None), now=NOW)
    assert ignored.disposition == "ignored"
    assert ignored.state.telemetry is not None and ignored.state.telemetry.validity == "no_estimate"


def test_staleness_is_derived_from_measurement_age() -> None:
    presence = _presence()
    presence.submit(_sample(), now=NOW)
    fresh = presence.latest("embodiment:microduck-01", now=NOW)
    assert fresh.status.fresh and fresh.status.valid
    later = NOW + timedelta(seconds=2.1)
    stale = presence.latest("embodiment:microduck-01", now=later)
    assert stale.status.stale and stale.status.valid and not stale.status.fresh
    ahead = derive_telemetry_status(fresh.telemetry, now=NOW - timedelta(seconds=5), stale_after_s=2)
    assert ahead.valid and not ahead.fresh and not ahead.stale and not ahead.unavailable


def test_unknown_embodiment_and_space_are_rejected() -> None:
    presence = _presence()
    unknown_body = _sample()
    unknown_body["embodiment_id"] = "embodiment:other"
    with pytest.raises(TelemetryAdmissionError, match="not registered") as missing_body:
        presence.submit(unknown_body, now=NOW)
    assert missing_body.value.code == "unknown_embodiment"
    with pytest.raises(TelemetryAdmissionError) as lookup:
        presence.latest("embodiment:other", now=NOW)
    assert lookup.value.code == "unknown_embodiment"
    unknown_space = _sample(space="space:garage")
    with pytest.raises(TelemetryAdmissionError, match="space") as missing_space:
        presence.submit(unknown_space, now=NOW)
    assert missing_space.value.code == "unknown_space"
    lost = _sample(validity="no_estimate", space="space:garage")
    with pytest.raises(TelemetryAdmissionError) as lost_space:
        presence.submit(lost, now=NOW)
    assert lost_space.value.code == "unknown_space"
    assert presence.latest("embodiment:microduck-01", now=NOW).telemetry is None


def test_node_files_register_valid_ids_and_skip_missing_files(tmp_path: Path) -> None:
    spaces = tmp_path / "space.json"
    spaces.write_text(json.dumps([{"id": "space:kitchen"}, {"id": "space:hall"}]), encoding="utf-8")
    bodies = tmp_path / "embodiment.json"
    bodies.write_text(json.dumps([{
        "id": "embodiment:microduck-01", "name": "MicroDuck",
        "local_frame": {"forward": "+x", "left": "+y", "up": "+z"},
        "geometry": {"box": {"length_m": 0.32, "width_m": 0.24, "height_m": 0.18,
                             "center": {"x": 0.04, "y": 0.0, "z": 0.09}}},
    }]), encoding="utf-8")
    assert record_ids_from_node_file(spaces, "space") == {"space:kitchen", "space:hall"}
    assert embodiment_ids_from_node_file(bodies) == {"embodiment:microduck-01"}
    assert record_ids_from_node_file(tmp_path / "missing.json", "space") == frozenset()
    assert embodiment_ids_from_node_file(tmp_path / "missing.json") == frozenset()
    spaces.write_text(json.dumps([{"id": "room:kitchen"}]), encoding="utf-8")
    with pytest.raises(SpatialContractError):
        record_ids_from_node_file(spaces, "space")


def test_http_submit_and_lookup() -> None:
    from home_cortex.agents.embodiments import EmbodimentCatalog, EmbodimentConnections
    from home_cortex.agents.session import EmbodimentSession
    from home_cortex.spatial.embodiment import embodiment_from_mapping

    previous = getattr(app.state, "embodiment_presence", None)
    previous_session = getattr(app.state, "embodiment_session", None)
    previous_settings = getattr(app.state, "settings", None)
    presence = _presence()
    catalog = EmbodimentCatalog([embodiment_from_mapping({
        "id": "embodiment:microduck-01", "name": "MicroDuck", "agent_id": "agent:butler",
        "capabilities": ["vision.observe"],
        "local_frame": {"forward": "+x", "left": "+y", "up": "+z"},
        "geometry": {"box": {"length_m": 0.32, "width_m": 0.24, "height_m": 0.18,
                             "center": {"x": 0.0, "y": 0.0, "z": 0.09}}},
    })])
    protocol = EmbodimentSession(catalog, EmbodimentConnections(catalog), presence)
    opened = protocol.register("embodiment:microduck-01", ["vision.observe"], now=NOW)
    app.state.embodiment_presence = presence
    app.state.embodiment_session = protocol
    app.state.settings = type("Settings", (), {"cortex_api_key": "test-cortex-key"})()
    client = TestClient(app, headers={
        "Authorization": "Bearer test-cortex-key",
        "X-Embodiment-Session-ID": opened["session"]["session_id"],
    })
    path = "/v1/embodiments/embodiment:microduck-01/telemetry"
    try:
        denied = client.post(path, headers={"Authorization": ""}, json=_sample())
        assert denied.status_code == 401
        created = client.post(path, json=_sample())
        assert created.status_code == 200
        assert created.json()["telemetry"]["disposition"] == "accepted"
        assert created.json()["telemetry"]["state"]["status"]["valid"] is True
        conflict = client.post(path, json=_sample(x=4))
        assert conflict.status_code == 409
        assert conflict.json()["error"]["code"] == "conflicting_sample"
        unknown_space = client.post(path, json=_sample(offset_s=4, space="space:garage"))
        assert unknown_space.status_code == 422
        assert unknown_space.json()["error"]["code"] == "unknown_space"
        older = client.post(path, json=_sample(offset_s=-1, x=8))
        assert older.status_code == 200
        assert older.json()["telemetry"]["disposition"] == "ignored"
        assert older.json()["telemetry"]["state"]["telemetry"]["transform"]["x"]["value"] == 1.372
        latest = client.get(path)
        assert latest.status_code == 200
        assert latest.json()["telemetry"]["measured_at"] == MEASURED
        missing = client.get("/v1/embodiments/embodiment:other/telemetry")
        assert missing.status_code == 404
        assert missing.json()["error"]["code"] == "unknown_embodiment"
        bad = _sample(offset_s=3)
        bad["transform"]["yaw"]["p95"] = -0.01
        rejected = client.post(path, json=bad)
        assert rejected.status_code == 422
        assert client.get(path).json()["telemetry"]["transform"]["x"]["value"] == 1.372
    finally:
        client.close()
        if previous is None:
            del app.state.embodiment_presence
        else:
            app.state.embodiment_presence = previous
        if previous_session is None:
            if hasattr(app.state, "embodiment_session"):
                del app.state.embodiment_session
        else:
            app.state.embodiment_session = previous_session
        if previous_settings is None:
            if hasattr(app.state, "settings"):
                del app.state.settings
        else:
            app.state.settings = previous_settings
