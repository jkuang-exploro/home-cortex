"""Persistent agent/body association, capabilities, and separate runtime presence."""
from pathlib import Path

import pytest

from home_cortex.agents import get_agent, get_agent_by_entity_id
from home_cortex.agents.embodiments import EmbodimentCatalog, EmbodimentConnections
from home_cortex.spatial.embodiment import embodiment_from_mapping
from home_cortex.spatial.primitives import SpatialContractError


def _body(key: str, capabilities: list[str]) -> dict:
    return {
        "id": f"embodiment:{key}", "name": key,
        "geometry": {"box": {"length_m": 0.3, "width_m": 0.2, "height_m": 0.1,
                             "center": {"x": 0, "y": 0, "z": 0.05}}},
        "local_frame": {"forward": "+x", "left": "+y", "up": "+z"},
        "capabilities": capabilities,
    }


def _catalog() -> EmbodimentCatalog:
    return EmbodimentCatalog((
        embodiment_from_mapping(_body("duck", ["mobility.move", "vision.observe"])),
        embodiment_from_mapping(_body("humanoid", ["mobility.move", "manipulation.grasp"])),
        embodiment_from_mapping(_body("speaker", ["audio.speak"])),
    ))


def test_registry_identity_is_independent_of_body_and_tools() -> None:
    agent = get_agent("steward")
    assert agent.entity_id == "agent:butler"
    assert get_agent_by_entity_id("agent:butler") is agent
    assert agent.display_name == "老管家"
    assert "mobility.move" not in agent.allowed_tools
    assert agent.id != "embodiment:duck"


def test_one_agent_can_own_multiple_bodies_and_capability_queries() -> None:
    catalog = _catalog().associate("embodiment:duck", "agent:butler")
    catalog = catalog.associate("embodiment:humanoid", "agent:butler")
    assert [body.id for body in catalog.bodies_for_agent("agent:butler")] == [
        "embodiment:duck", "embodiment:humanoid",
    ]
    assert [body.id for body in catalog.with_capability("mobility.move")] == [
        "embodiment:duck", "embodiment:humanoid",
    ]
    assert [body.id for body in catalog.named("duck")] == ["embodiment:duck"]
    assert catalog.get("embodiment:speaker").agent_id is None


def test_one_body_cannot_have_two_controlling_agents() -> None:
    catalog = EmbodimentCatalog(
        [embodiment_from_mapping(_body("duck", []))],
        known_agent_ids={"agent:butler", "agent:other"},
    ).associate("embodiment:duck", "agent:butler")
    with pytest.raises(SpatialContractError, match="already controlled"):
        catalog.associate("embodiment:duck", "agent:other")
    assert catalog.get("embodiment:duck").agent_id == "agent:butler"
    transferred = catalog.unassign("embodiment:duck").associate(
        "embodiment:duck", "agent:other"
    )
    assert transferred.get("embodiment:duck").agent_id == "agent:other"
    assert catalog.get("embodiment:duck").agent_id == "agent:butler"


def test_catalog_rejects_association_to_unregistered_agent() -> None:
    body = _body("duck", [])
    body["agent_id"] = "agent:unknown"
    with pytest.raises(SpatialContractError, match="unknown controlling agent"):
        EmbodimentCatalog([embodiment_from_mapping(body)])


def test_unassign_and_persisted_association_survive_disconnect(tmp_path: Path) -> None:
    path = tmp_path / "embodiment.json"
    assigned = _catalog().associate("embodiment:duck", "agent:butler")
    assigned.save_node_file(path)
    reloaded = EmbodimentCatalog.from_node_file(path)
    connections = EmbodimentConnections(reloaded)
    assert not connections.is_currently_embodied("agent:butler")
    connections.connect("embodiment:duck", available_capabilities={"vision.observe"})
    assert connections.is_currently_embodied("agent:butler")
    assert connections.active_capabilities("embodiment:duck") == {"vision.observe"}
    connections.disconnect("embodiment:duck")
    assert not connections.is_currently_embodied("agent:butler")
    assert EmbodimentCatalog.from_node_file(path).get("embodiment:duck").agent_id == "agent:butler"
    unassigned = reloaded.unassign("embodiment:duck")
    unassigned.save_node_file(path)
    assert EmbodimentCatalog.from_node_file(path).get("embodiment:duck").agent_id is None


def test_runtime_capability_must_be_configured_and_does_not_change_record() -> None:
    catalog = _catalog().associate("embodiment:duck", "agent:butler")
    connections = EmbodimentConnections(catalog)
    with pytest.raises(SpatialContractError, match="must be configured"):
        connections.connect("embodiment:duck", available_capabilities={"audio.speak"})
    assert not connections.is_connected("embodiment:duck")
    connections.connect("embodiment:duck", available_capabilities=set())
    assert connections.is_currently_embodied("agent:butler")
    assert connections.active_capabilities("embodiment:duck") == frozenset()
    assert catalog.get("embodiment:duck").capabilities == (
        "mobility.move", "vision.observe",
    )


@pytest.mark.parametrize("capabilities", [
    ["move"], ["mobility.move", "mobility.move"], ["Mobility.move"],
])
def test_invalid_or_duplicate_capability_names(capabilities: list[str]) -> None:
    with pytest.raises(SpatialContractError):
        embodiment_from_mapping(_body("duck", capabilities))


def test_capability_namespace_is_open_vocabulary() -> None:
    assert embodiment_from_mapping(_body("duck", ["example.custom_action"])).capabilities == (
        "example.custom_action",
    )
