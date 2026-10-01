"""Read projection and physical availability keep identity, session, pose apart."""
from datetime import datetime, timezone

import pytest

from home_cortex.agents.embodiments import EmbodimentCatalog, EmbodimentConnections
from home_cortex.agents.presence import EmbodimentDirectory
from home_cortex.agents.registration import CatalogIdentityReader
from home_cortex.spatial.embodiment import embodiment_from_mapping
from home_cortex.spatial.presence import EmbodimentPresence


BODY = "embodiment:microduck-01"


def directory():
    body = embodiment_from_mapping({
        "id": BODY, "name": "MicroDuck", "agent_id": "agent:butler",
        "capabilities": ["mobility.move", "vision.observe"],
        "local_frame": {"forward": "+x", "left": "+y", "up": "+z"},
        "geometry": {"box": {"length_m": 0.32, "width_m": 0.24, "height_m": 0.18,
                             "center": {"x": 0, "y": 0, "z": 0.09}}},
    })
    other = embodiment_from_mapping({
        "id": "embodiment:unlinked", "name": "Other",
        "capabilities": [],
        "local_frame": {"forward": "+x", "left": "+y", "up": "+z"},
        "geometry": {"box": {"length_m": 0.2, "width_m": 0.2, "height_m": 0.2,
                             "center": {"x": 0, "y": 0, "z": 0}}},
    })
    catalog = EmbodimentCatalog([body, other])
    connections = EmbodimentConnections(catalog)
    presence = EmbodimentPresence(embodiment_ids=catalog.embodiment_ids,
                                  space_ids=["space:kitchen"], stale_after_s=2)
    return EmbodimentDirectory(CatalogIdentityReader(catalog), connections, presence)


@pytest.mark.asyncio
async def test_unlinked_offline_online_and_missing_telemetry_are_distinct() -> None:
    service = directory()
    assert [item["id"] for item in await service.list()] == [BODY, "embodiment:unlinked"]
    assert (await service.get("embodiment:unlinked"))["state"] == "unlinked"
    offline = await service.get(BODY)
    assert offline["state"] == "linked_offline"
    assert offline["agent"] == {"id": "agent:butler", "name": "老管家"}
    assert offline["telemetry"]["available"] is False
    assert offline["geometry"]["box"]["length_m"] == 0.32
    service.connections.connect(BODY, available_capabilities=["vision.observe"])
    assert (await service.action_availability(
        "agent:other", BODY, "vision.observe")).code == "embodiment_not_linked"
    online = await service.get(BODY)
    assert online["state"] == "linked_online"
    assert online["telemetry"]["available"] is False
    assert online["capabilities"] == [
        {"name": "mobility.move", "supported": True, "available": False},
        {"name": "vision.observe", "supported": True, "available": True},
    ]
    assert "session_id" not in online["runtime"]
    service.connections.disconnect(BODY)
    assert (await service.get(BODY))["state"] == "linked_offline"
    service.connections.connect(BODY, available_capabilities=["mobility.move"])
    refreshed = await service.get(BODY)
    assert refreshed["id"] == BODY
    assert refreshed["agent"]["id"] == "agent:butler"


@pytest.mark.asyncio
async def test_telemetry_is_independent_and_action_gate_checks_each_condition() -> None:
    service = directory()
    assert (await service.action_availability(
        "agent:butler", "embodiment:missing", "mobility.move")).code == "unknown_embodiment"
    assert (await service.action_availability(
        "agent:butler", "embodiment:unlinked", "mobility.move")).code == "embodiment_not_linked"
    assert (await service.action_availability(
        "agent:butler", BODY, "mobility.move")).code == "embodiment_offline"
    service.connections.connect(BODY, available_capabilities=["vision.observe"])
    assert (await service.action_availability(
        "agent:butler", BODY, "audio.speak")).code == "unsupported_capability"
    assert (await service.action_availability(
        "agent:butler", BODY, "mobility.move")).code == "capability_unavailable"
    assert (await service.action_availability("agent:butler", BODY, "vision.observe")).available
    now = datetime.now(timezone.utc)
    service.presence.submit({"embodiment_id": BODY, "space_id": None,
                             "measured_at": now.isoformat(), "validity": "no_estimate",
                             "transform": None}, now=now)
    viewed = await service.get(BODY)
    assert viewed["connected"] is True
    assert viewed["telemetry"]["available"] is False
    service.connections.disconnect(BODY)
    assert (await service.get(BODY))["telemetry"]["available"] is False
    assert (await service.action_availability(
        "agent:butler", BODY, "vision.observe")).code == "embodiment_offline"


@pytest.mark.asyncio
async def test_fresh_pose_is_exposed_without_controlling_connection_state() -> None:
    service = directory()
    service.connections.connect(BODY, available_capabilities=[])
    measured = datetime.now(timezone.utc)
    service.presence.submit({
        "embodiment_id": BODY, "space_id": "space:kitchen",
        "measured_at": measured.isoformat(), "validity": "valid",
        "transform": {name: {"value": value, "p95": 0.02}
                      for name, value in (
                          ("x", 1.0), ("y", 2.0), ("z", 0.0),
                          ("yaw", 0.0), ("pitch", 0.0), ("roll", 0.0))},
    }, now=measured)
    view = await service.get(BODY)
    assert view["telemetry"]["available"] is True
    assert view["telemetry"]["space_id"] == "space:kitchen"
    assert view["telemetry"]["estimate"]["transform"]["x"]["p95"] == 0.02
    service.connections.disconnect(BODY)
    disconnected = await service.get(BODY)
    assert disconnected["connected"] is False
    assert disconnected["telemetry"]["available"] is False
