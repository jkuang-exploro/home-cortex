"""Simulated client transitions projected into the GUI's read contracts."""
from datetime import datetime, timedelta, timezone

from home_cortex.agents.embodiments import EmbodimentCatalog, EmbodimentConnections
from home_cortex.agents.presence import EmbodimentDirectory
from home_cortex.agents.session import EmbodimentSession
from home_cortex.spatial.embodiment import embodiment_from_mapping
from home_cortex.spatial.presence import EmbodimentPresence
from scripts.maintenance.embodiment_client import SimulatedEmbodimentClient


BODY = "embodiment:microduck-01"


def test_directory_follows_simulated_client_without_changing_body_identity() -> None:
    body = embodiment_from_mapping({
        "id": BODY, "name": "MicroDuck",
        "capabilities": ["mobility.move", "vision.observe", "audio.speak"],
        "local_frame": {"forward": "+x", "left": "+y", "up": "+z"},
        "geometry": {"box": {"length_m": 0.32, "width_m": 0.24, "height_m": 0.18,
                             "center": {"x": 0.04, "y": 0, "z": 0.09}}},
    })
    unlinked = EmbodimentCatalog([body])
    assert unlinked.get(BODY).agent_id is None

    catalog = unlinked.associate(BODY, "agent:butler")
    connections = EmbodimentConnections(catalog)
    presence = EmbodimentPresence(
        embodiment_ids=catalog.embodiment_ids, space_ids=["space:kitchen"], stale_after_s=2,
    )
    directory = EmbodimentDirectory(catalog, connections, presence)
    client = SimulatedEmbodimentClient(EmbodimentSession(catalog, connections, presence), BODY)
    started = datetime.now(timezone.utc)

    offline = directory.get(BODY)
    assert offline["linked"] and not offline["connected"]
    assert offline["agent"] == {"id": "agent:butler", "name": "老管家"}
    assert offline["telemetry"]["available"] is False

    first = client.register(["mobility.move", "vision.observe", "audio.speak"], now=started)
    online = directory.get(BODY)
    assert online["connected"] and not online["telemetry"]["available"]
    assert all(item["available"] for item in online["capabilities"])

    client.send_telemetry({
        "space_id": "space:kitchen", "measured_at": started.isoformat(),
        "validity": "valid",
        "transform": {name: {"value": value, "p95": 0.02}
                      for name, value in (("x", 1), ("y", 2), ("z", 0),
                                          ("yaw", 0), ("pitch", 0), ("roll", 0))},
    }, now=started)
    assert directory.get(BODY)["telemetry"]["available"] is True

    later = started + timedelta(milliseconds=100)
    client.send_no_estimate(measured_at=later.isoformat(),
                            space_id="space:kitchen", now=later)
    no_pose = directory.get(BODY)
    assert no_pose["connected"] and not no_pose["telemetry"]["available"]
    assert no_pose["telemetry"]["estimate"]["validity"] == "no_estimate"

    client.change_capabilities(["vision.observe"], now=later)
    current = {item["name"]: item for item in directory.get(BODY)["capabilities"]}
    assert current["mobility.move"]["supported"] and not current["mobility.move"]["available"]
    assert current["vision.observe"]["available"]

    client.disconnect(now=later)
    disconnected = directory.get(BODY)
    assert not disconnected["connected"]
    assert directory.action_availability("agent:butler", BODY, "vision.observe").code == "embodiment_offline"

    second = client.reconnect(["mobility.move"], now=later)
    reconnected = directory.get(BODY)
    assert second["session"]["session_id"] != first["session"]["session_id"]
    assert reconnected["id"] == offline["id"] == BODY
    assert reconnected["agent"] == offline["agent"]
    assert reconnected["connected"]
    assert directory.action_availability("agent:butler", BODY, "mobility.move").available
