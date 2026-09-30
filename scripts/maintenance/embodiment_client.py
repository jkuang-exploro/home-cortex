"""Speak the canonical embodiment session protocol. This client has no sensors."""
from __future__ import annotations

import argparse
import json
from collections.abc import Collection, Mapping
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from home_cortex.agents.embodiments import EmbodimentCatalog, EmbodimentConnections
from home_cortex.agents.session import EmbodimentSession
from home_cortex.spatial.embodiment import embodiment_from_mapping
from home_cortex.spatial.presence import EmbodimentPresence


class SimulatedEmbodimentClient:
    """Engineering stand-in that sends canonical session messages, not sensors."""

    def __init__(self, session: EmbodimentSession, embodiment_id: str) -> None:
        self.session = session
        self.embodiment_id = embodiment_id
        self.session_id: str | None = None

    def register(self, available_capabilities: Collection[str], *, now: datetime | None = None):
        result = self.session.register(self.embodiment_id, available_capabilities, now=now)
        self.session_id = result["session"]["session_id"]
        return result

    def send_telemetry(self, sample: Mapping[str, Any], *, now: datetime | None = None):
        payload = dict(sample)
        payload["embodiment_id"] = self.embodiment_id
        return self.session.submit_telemetry(payload, session_id=self.session_id, now=now)

    def send_no_estimate(
        self, *, measured_at: str, space_id: str | None = None, now: datetime | None = None,
    ):
        return self.send_telemetry({
            "space_id": space_id, "measured_at": measured_at,
            "validity": "no_estimate", "transform": None,
        }, now=now)

    def change_capabilities(self, available_capabilities: Collection[str], *, now: datetime | None = None):
        return self.session.update_capabilities(
            self.embodiment_id, available_capabilities,
            session_id=self.session_id, now=now,
        )

    def heartbeat(self, *, now: datetime | None = None):
        return self.session.heartbeat(self.embodiment_id, session_id=self.session_id, now=now)

    def disconnect(self, *, now: datetime | None = None):
        return self.session.disconnect(self.embodiment_id, session_id=self.session_id, now=now)

    def reconnect(self, available_capabilities: Collection[str], *, now: datetime | None = None):
        return self.register(available_capabilities, now=now)


def run_demo(*, embodiment_id: str = "embodiment:microduck-01") -> list[dict[str, Any]]:
    """In-memory boot sequence for one known body. Creates no persistent record."""
    started = datetime(2026, 9, 28, 18, 0, tzinfo=timezone.utc)
    body = embodiment_from_mapping({
        "id": embodiment_id, "name": "MicroDuck", "agent_id": "agent:butler",
        "capabilities": ["audio.speak", "mobility.move", "vision.observe"],
        "local_frame": {"forward": "+x", "left": "+y", "up": "+z"},
        "geometry": {"box": {"length_m": 0.32, "width_m": 0.24, "height_m": 0.18,
                             "center": {"x": 0.04, "y": 0.0, "z": 0.09}}},
    })
    catalog = EmbodimentCatalog([body])
    presence = EmbodimentPresence(
        embodiment_ids=catalog.embodiment_ids, space_ids=["space:kitchen"], stale_after_s=2,
    )
    client = SimulatedEmbodimentClient(
        EmbodimentSession(catalog, EmbodimentConnections(catalog), presence), embodiment_id,
    )
    available = ["mobility.move", "vision.observe"]
    return [
        client.register(available, now=started),
        client.send_telemetry(_pose(embodiment_id, started), now=started + timedelta(seconds=1)),
        client.send_no_estimate(measured_at=(started + timedelta(seconds=2)).isoformat(), now=started + timedelta(seconds=2)),
        client.change_capabilities(["vision.observe"], now=started + timedelta(seconds=3)),
        client.heartbeat(now=started + timedelta(seconds=4)),
        client.disconnect(now=started + timedelta(seconds=5)),
        client.reconnect(available, now=started + timedelta(seconds=6)),
    ]


def _pose(embodiment_id: str, measured_at: datetime) -> dict[str, Any]:
    return {
        "embodiment_id": embodiment_id, "space_id": "space:kitchen",
        "measured_at": measured_at.isoformat(), "validity": "valid",
        "transform": {name: {"value": value, "p95": uncertainty}
                      for name, value, uncertainty in (
                          ("x", 1.0, 0.02), ("y", 2.0, 0.02), ("z", 0.0, 0.01),
                          ("yaw", 0.0, 0.01), ("pitch", 0.0, 0.01), ("roll", 0.0, 0.01),
                      )},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Register a known embodiment, exchange telemetry, and disconnect.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("demo", help="Run the in-memory lifecycle for embodiment:microduck-01")
    remote = commands.add_parser("register", help="Open a runtime session on a Home Cortex API")
    _remote_arguments(remote)
    remote.add_argument("--available", action="append", default=[], help="Capability available now")
    arguments = parser.parse_args(argv)
    if arguments.command == "demo":
        print(json.dumps(run_demo(), ensure_ascii=False, indent=2))
        return 0
    payload = json.dumps({"available_capabilities": arguments.available}).encode()
    request = Request(
        f"{arguments.base_url.rstrip('/')}/v1/embodiments/{arguments.embodiment}/session",
        data=payload,
        headers={
            "Authorization": f"Bearer {arguments.api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=30) as response:
            print(response.read().decode())
    except HTTPError as error:
        print(error.read().decode())
        return 1
    return 0


def _remote_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--embodiment", required=True)


if __name__ == "__main__":
    raise SystemExit(main())
