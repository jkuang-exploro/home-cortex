"""Runtime session lifecycle for a simulated embodiment client."""
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from home_cortex.agents import get_agent
from home_cortex.agents.embodiments import EmbodimentCatalog, EmbodimentConnections, SessionProtocolError
from home_cortex.agents.registration import CatalogIdentityReader, EmbodimentRegistration
from home_cortex.agents.session import EmbodimentSession
from home_cortex.api.app import app
from home_cortex.spatial.embodiment import embodiment_from_mapping
from home_cortex.spatial.presence import EmbodimentPresence
from scripts.maintenance.embodiment_client import SimulatedEmbodimentClient, main, run_demo


NOW = datetime(2026, 9, 28, 18, 0, tzinfo=timezone.utc)
BODY = "embodiment:microduck-01"


def _body() -> dict:
    return {
        "id": BODY, "name": "MicroDuck", "agent_id": "agent:butler",
        "capabilities": ["audio.speak", "mobility.move", "vision.observe"],
        "local_frame": {"forward": "+x", "left": "+y", "up": "+z"},
        "geometry": {"box": {"length_m": 0.32, "width_m": 0.24, "height_m": 0.18,
                             "center": {"x": 0.04, "y": 0.0, "z": 0.09}}},
    }


def _stack():
    catalog = EmbodimentCatalog([embodiment_from_mapping(_body())])
    presence = EmbodimentPresence(
        embodiment_ids=catalog.embodiment_ids, space_ids=["space:kitchen"], stale_after_s=2,
    )
    session = EmbodimentSession(catalog, EmbodimentConnections(catalog), presence)
    return catalog, session, SimulatedEmbodimentClient(session, BODY)


def _pose(*, offset_s: float = 0, x: float = 1.0) -> dict:
    return {
        "space_id": "space:kitchen",
        "measured_at": (NOW + timedelta(seconds=offset_s)).isoformat(),
        "validity": "valid",
        "transform": {
            name: {"value": value, "p95": 0.02}
            for name, value in (("x", x), ("y", 2.0), ("z", 0.0), ("yaw", 0.0), ("pitch", 0.0), ("roll", 0.0))
        },
    }


def test_first_registration_binds_the_existing_agent() -> None:
    catalog, _, client = _stack()
    opened = client.register(["mobility.move", "vision.observe"], now=NOW)
    assert opened["disposition"] == "connected"
    assert opened["embodiment_id"] == BODY
    assert opened["agent_id"] == "agent:butler"
    assert opened["session"]["online"] is True
    assert opened["session"]["connected_at"] == opened["session"]["last_seen"]
    assert opened["session"]["available_capabilities"] == ["mobility.move", "vision.observe"]
    assert "audio.speak" in opened["configured_capabilities"]
    assert catalog.get(BODY).agent_id == "agent:butler"
    assert catalog.embodiment_ids == {BODY}
    assert get_agent("steward").display_name == "老管家"
    assert get_agent("steward").entity_id == "agent:butler"


def test_unknown_embodiment_does_not_allocate_another_identity() -> None:
    catalog, session, _ = _stack()
    stranger = SimulatedEmbodimentClient(session, "embodiment:microduck-02")
    with pytest.raises(SessionProtocolError, match="unknown embodiment") as error:
        stranger.register(["vision.observe"], now=NOW)
    assert error.value.code == "unknown_embodiment"
    assert catalog.embodiment_ids == {BODY}
    assert catalog.get(BODY).id == BODY


def test_duplicate_connection_replaces_the_runtime_session() -> None:
    _, _, client = _stack()
    first = client.register(["vision.observe"], now=NOW)
    second = client.register(["mobility.move"], now=NOW + timedelta(seconds=1))
    assert second["disposition"] == "replaced"
    assert second["embodiment_id"] == first["embodiment_id"]
    assert second["session"]["session_id"] != first["session"]["session_id"]
    assert second["session"]["online"] is True
    assert second["session"]["available_capabilities"] == ["mobility.move"]
    assert second["agent_id"] == "agent:butler"


def test_replaced_client_cannot_mutate_the_new_session() -> None:
    _, protocol, old_client = _stack()
    old_client.register(["vision.observe"], now=NOW)
    new_client = SimulatedEmbodimentClient(protocol, BODY)
    new_client.register(["mobility.move"], now=NOW + timedelta(seconds=1))
    with pytest.raises(SessionProtocolError) as stale:
        old_client.send_telemetry(_pose(offset_s=2), now=NOW + timedelta(seconds=2))
    assert stale.value.code == "stale_session"
    assert protocol.presence.latest(BODY, now=NOW + timedelta(seconds=2)).telemetry is None
    with pytest.raises(SessionProtocolError) as disconnect:
        old_client.disconnect(now=NOW + timedelta(seconds=2))
    assert disconnect.value.code == "stale_session"
    with pytest.raises(SessionProtocolError) as heartbeat:
        old_client.heartbeat(now=NOW + timedelta(seconds=2))
    assert heartbeat.value.code == "stale_session"
    with pytest.raises(SessionProtocolError) as capabilities:
        old_client.change_capabilities(["vision.observe"], now=NOW + timedelta(seconds=2))
    assert capabilities.value.code == "stale_session"
    assert new_client.heartbeat(now=NOW + timedelta(seconds=3))["session"]["online"]


def test_session_ids_differ_across_runtime_restarts() -> None:
    _, _, first = _stack()
    _, _, restarted = _stack()
    assert first.register(["vision.observe"], now=NOW)["session"]["session_id"] != (
        restarted.register(["vision.observe"], now=NOW)["session"]["session_id"]
    )


def test_telemetry_requires_registration_and_then_updates_last_seen() -> None:
    _, session, client = _stack()
    with pytest.raises(SessionProtocolError, match="online runtime session") as rejected:
        client.send_telemetry(_pose(), now=NOW)
    assert rejected.value.code == "session_offline"
    assert session.presence.latest(BODY, now=NOW).telemetry is None
    client.register(["vision.observe"], now=NOW)
    accepted = client.send_telemetry(_pose(offset_s=1), now=NOW + timedelta(seconds=1))
    assert accepted["telemetry"]["disposition"] == "accepted"
    assert accepted["presence"]["session"]["last_seen"] == (NOW + timedelta(seconds=1)).isoformat()
    assert accepted["presence"]["session"]["connected_at"] == NOW.isoformat()
    missing = client.send_no_estimate(
        measured_at=(NOW + timedelta(seconds=2)).isoformat(), now=NOW + timedelta(seconds=2),
    )
    assert missing["telemetry"]["state"]["telemetry"]["validity"] == "no_estimate"
    assert missing["presence"]["session"]["online"] is True


def test_capability_refresh_keeps_the_session_and_configured_body() -> None:
    catalog, _, client = _stack()
    opened = client.register(
        ["audio.speak", "mobility.move", "vision.observe"], now=NOW,
    )
    refreshed = client.change_capabilities(["vision.observe"], now=NOW + timedelta(seconds=2))
    assert refreshed["disposition"] == "capabilities"
    assert refreshed["session"]["session_id"] == opened["session"]["session_id"]
    assert refreshed["session"]["available_capabilities"] == ["vision.observe"]
    assert refreshed["session"]["connected_at"] == NOW.isoformat()
    assert catalog.get(BODY).capabilities == ("audio.speak", "mobility.move", "vision.observe")
    with pytest.raises(SessionProtocolError, match="must be configured") as error:
        client.change_capabilities(["manipulation.grasp"], now=NOW + timedelta(seconds=3))
    assert error.value.code == "capability_not_configured"
    assert client.session.connections.current(BODY).available_capabilities == frozenset({"vision.observe"})


def test_disconnect_and_reconnect_preserve_identity_and_agent() -> None:
    catalog, session, client = _stack()
    opened = client.register(["vision.observe"], now=NOW)
    offline = client.disconnect(now=NOW + timedelta(seconds=1))
    assert offline["disposition"] == "disconnected"
    assert offline["session"]["online"] is False
    assert offline["session"]["session_id"] == opened["session"]["session_id"]
    assert catalog.get(BODY).agent_id == "agent:butler"
    with pytest.raises(SessionProtocolError) as blocked:
        client.send_telemetry(_pose(offset_s=2), now=NOW + timedelta(seconds=2))
    assert blocked.value.code == "session_offline"
    online = client.reconnect(["mobility.move", "vision.observe"], now=NOW + timedelta(seconds=3))
    assert online["disposition"] == "connected"
    assert online["session"]["online"] is True
    assert online["session"]["session_id"] != opened["session"]["session_id"]
    assert online["embodiment_id"] == BODY
    assert online["agent_id"] == "agent:butler"
    assert session.catalog.embodiment_ids == {BODY}
    assert get_agent("steward").display_name == "老管家"
    assert get_agent_by_entity_still_butler()


def get_agent_by_entity_still_butler() -> bool:
    from home_cortex.agents import get_agent_by_entity_id
    return get_agent_by_entity_id("agent:butler").id == "steward"


def test_heartbeat_advances_last_seen_without_a_new_session() -> None:
    _, _, client = _stack()
    opened = client.register(["vision.observe"], now=NOW)
    beat = client.heartbeat(now=NOW + timedelta(seconds=5))
    assert beat["disposition"] == "heartbeat"
    assert beat["session"]["session_id"] == opened["session"]["session_id"]
    assert beat["session"]["connected_at"] == NOW.isoformat()
    assert beat["session"]["last_seen"] == (NOW + timedelta(seconds=5)).isoformat()


def test_demo_command_uses_one_body_and_the_existing_agent(capsys) -> None:
    steps = run_demo()
    assert steps[-1]["embodiment_id"] == BODY
    assert steps[-1]["agent_id"] == "agent:butler"
    assert steps[-1]["session"]["online"] is True
    assert steps[5]["session"]["online"] is False
    assert all(
        step.get("embodiment_id", step.get("presence", {}).get("embodiment_id")) == BODY
        for step in steps
    )
    assert main(["demo"]) == 0
    output = capsys.readouterr().out
    assert "embodiment:microduck-01" in output
    assert "embodiment:microduck-02" not in output
    assert "agent:butler" in output


def test_http_session_gates_telemetry() -> None:
    catalog, session, _ = _stack()
    previous_session = getattr(app.state, "embodiment_session", None)
    previous_registration = getattr(app.state, "embodiment_registration", None)
    previous_settings = getattr(app.state, "settings", None)
    app.state.embodiment_session = session
    app.state.embodiment_registration = EmbodimentRegistration(
        CatalogIdentityReader(catalog), session,
    )
    app.state.embodiment_catalog = catalog
    app.state.settings = type("Settings", (), {"cortex_api_key": "test-cortex-key"})()
    client = TestClient(app, headers={"Authorization": "Bearer test-cortex-key"})
    path = f"/v1/embodiments/{BODY}/session"
    telemetry = f"/v1/embodiments/{BODY}/telemetry"
    try:
        denied = client.post(path, headers={"Authorization": ""}, json={"available_capabilities": []})
        assert denied.status_code == 401
        chosen = client.post(path, json={
            "available_capabilities": ["vision.observe"], "agent_id": "agent:butler",
        })
        assert chosen.status_code == 422
        assert chosen.json()["error"]["code"] == "invalid_session"
        assert catalog.get(BODY).agent_id == "agent:butler"
        early = client.post(telemetry, json={**_pose(), "embodiment_id": BODY})
        assert early.status_code == 409
        assert early.json()["error"]["code"] == "session_offline"
        opened = client.post(path, json={"available_capabilities": ["vision.observe"]})
        assert opened.status_code == 200
        assert opened.json()["disposition"] == "connected"
        assert opened.json()["agent_id"] == "agent:butler"
        missing_session = client.post(telemetry, json={**_pose(offset_s=1), "embodiment_id": BODY})
        assert missing_session.status_code == 409
        assert missing_session.json()["error"]["code"] == "stale_session"
        client.headers["X-Embodiment-Session-ID"] = opened.json()["session"]["session_id"]
        posted = client.post(telemetry, json={**_pose(offset_s=1), "embodiment_id": BODY})
        assert posted.status_code == 200
        assert posted.json()["telemetry"]["disposition"] == "accepted"
        closed = client.delete(path)
        assert closed.status_code == 200
        assert closed.json()["session"]["online"] is False
        blocked = client.post(telemetry, json={**_pose(offset_s=2), "embodiment_id": BODY})
        assert blocked.status_code == 409
        again = client.post(path, json={"available_capabilities": ["mobility.move"]})
        assert again.status_code == 200
        assert again.json()["session"]["online"] is True
        assert again.json()["session"]["session_id"] != opened.json()["session"]["session_id"]
        stale_post = client.post(telemetry, json={**_pose(offset_s=3), "embodiment_id": BODY})
        assert stale_post.status_code == 409
        assert stale_post.json()["error"]["code"] == "stale_session"
        client.headers["X-Embodiment-Session-ID"] = again.json()["session"]["session_id"]
        unknown = client.post(
            "/v1/embodiments/embodiment:microduck-02/session",
            json={"available_capabilities": []},
        )
        assert unknown.status_code == 404
        assert catalog.embodiment_ids == {BODY}
    finally:
        client.close()
        if previous_session is None:
            if hasattr(app.state, "embodiment_session"):
                del app.state.embodiment_session
        else:
            app.state.embodiment_session = previous_session
        if previous_registration is None:
            if hasattr(app.state, "embodiment_registration"):
                del app.state.embodiment_registration
        else:
            app.state.embodiment_registration = previous_registration
        if previous_settings is None:
            if hasattr(app.state, "settings"):
                del app.state.settings
        else:
            app.state.settings = previous_settings
