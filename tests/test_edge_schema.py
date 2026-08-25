from pathlib import Path

import pytest

from home_cortex.edge_schema import EdgeSchemaRegistry, UnknownEdgeSchemaError

SCHEMA_DIR = Path(__file__).parents[1] / "schemas" / "edge"


def test_registry_loads_required_relationship_semantics() -> None:
    registry = EdgeSchemaRegistry.from_directory(SCHEMA_DIR)

    assert registry.get("spouse_of").symmetric is True
    assert registry.get("spouse_of").temporal is True
    assert registry.get("parent_of").symmetric is False
    assert registry.get("parent_of").inverse_name == "child_of"
    assert registry.get("lives_in").temporal is True
    assert registry.get("contained_in").symmetric is False
    assert registry.get("contained_in").temporal is False
    assert registry.get("contained_in").inverse_name == "contains"
    assert registry.get("contained_in").from_types == ("space",)
    assert registry.get("contained_in").to_types == ("location", "space")
    hosted_by = registry.get("hosted_by")
    assert hosted_by.symmetric is False
    assert hosted_by.temporal is False
    assert hosted_by.inverse_name == "hosts_space"
    assert hosted_by.from_types == ("space",)
    assert hosted_by.to_types == ("item",)
    assert hosted_by.unique_from is True
    assert registry.get("contained_in").unique_from is False
    located_in = registry.get("located_in")
    assert located_in.from_types == ("item",)
    assert located_in.to_types == ("space",)
    assert located_in.unique_from is True


def test_registry_resolves_inverse_without_registering_duplicate_schema() -> None:
    registry = EdgeSchemaRegistry.from_directory(SCHEMA_DIR)

    resolved = registry.resolve("child_of")

    assert resolved.schema.id == "parent_of"
    assert resolved.inverse is True
    assert "child_of" not in registry.relationship_names

    contains = registry.resolve("contains")
    assert contains.schema.id == "contained_in"
    assert contains.inverse is True
    assert "contains" not in registry.relationship_names

    hosts_space = registry.resolve("hosts_space")
    assert hosts_space.schema.id == "hosted_by"
    assert hosts_space.inverse is True
    assert "hosts_space" not in registry.relationship_names


def test_unknown_relationship_fails_cleanly() -> None:
    registry = EdgeSchemaRegistry.from_directory(SCHEMA_DIR)

    with pytest.raises(UnknownEdgeSchemaError, match="Unknown relationship"):
        registry.get("invented_relation")


def test_registry_validates_endpoint_types() -> None:
    registry = EdgeSchemaRegistry.from_directory(SCHEMA_DIR)

    registry.validate_endpoints("lives_in", "person", "location")
    with pytest.raises(ValueError, match="Invalid lives_in endpoints"):
        registry.validate_endpoints("lives_in", "location", "person")
    registry.validate_endpoints("contained_in", "space", "location")
    registry.validate_endpoints("contained_in", "space", "space")
    with pytest.raises(ValueError, match="Invalid contained_in endpoints"):
        registry.validate_endpoints("contained_in", "location", "space")
    with pytest.raises(ValueError, match="Invalid contained_in endpoints"):
        registry.validate_endpoints("contained_in", "person", "location")
    registry.validate_endpoints("hosted_by", "space", "item")
    with pytest.raises(ValueError, match="Invalid hosted_by endpoints"):
        registry.validate_endpoints("hosted_by", "item", "space")
