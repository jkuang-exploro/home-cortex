"""Persistent space and located_in pose contracts; no localization."""
import json
import math
from pathlib import Path
from shutil import copytree
from typing import Any

import pytest
from surrealdb import AsyncSurreal

from home_cortex.edge_schema import EdgeSchemaRegistry
from home_cortex.export import export_directory
from home_cortex.ingestion import ingest_directory
from home_cortex.spatial.contracts import (
    LENGTH_UNIT,
    SpatialContractError,
    apply_located_in_pose,
    apply_space_spatial_fields,
    located_in_endpoints_allowed,
    reject_non_space_spatial_fields,
)

ROOT = Path(__file__).parents[1]
STATIC_TEST_DATA = Path(__file__).parent / "static_test_data"
PRODUCTION_DATA = ROOT / "data"


class MemoryDatabase:
    def __init__(self) -> None:
        self.client = AsyncSurreal("mem://")

    async def connect(self) -> None:
        await self.client.connect()
        await self.client.use("test", "spatial")

    async def close(self) -> None:
        await self.client.close()

    async def upsert(self, record: Any, data: dict[str, Any]) -> Any:
        return await self.client.upsert(record, data)

    async def query(
        self,
        statement: str,
        variables: dict[str, Any] | None = None,
    ) -> Any:
        return await self.client.query(statement, variables or {})


def _space(**fields: Any) -> dict[str, Any]:
    return {"id": "space:kitchen", "name": {"en": "Kitchen"}, **fields}


def test_space_coordinate_metadata_canonicalizes_si_unit() -> None:
    record = _space(
        coordinate={
            "unit": "meter",
            "origin": [0, 0, 0],
            "x_axis": [1, 0, 0],
            "y_axis": [0, 1, 0],
            "z_axis": "up",
        }
    )
    apply_space_spatial_fields(record, source="space.json")
    assert record["coordinate"]["unit"] == LENGTH_UNIT
    assert record["coordinate"]["origin"] == [0, 0, 0]
    assert record["coordinate"]["z_axis"] == "up"


def test_space_navigable_and_accessible_booleans() -> None:
    record = _space(navigable=True, accessible=False)
    apply_space_spatial_fields(record, source="space.json")
    assert record["navigable"] is True
    assert record["accessible"] is False


def test_legacy_space_without_spatial_metadata_is_unchanged() -> None:
    record = _space(space_type="room")
    apply_space_spatial_fields(record, source="space.json")
    assert set(record) == {"id", "name", "space_type"}


def test_located_in_without_pose_is_unchanged() -> None:
    record = {"from": "item:mug", "to": "space:kitchen"}
    apply_located_in_pose(record, target_type="space", source="located_in.json")
    assert record == {"from": "item:mug", "to": "space:kitchen"}


def test_located_in_with_position() -> None:
    record = {
        "from": "item:mug",
        "to": "space:kitchen",
        "position": {"x": 1.42, "y": 0.83, "z": 0.91},
    }
    apply_located_in_pose(record, target_type="space", source="located_in.json")
    assert record["position"] == {"x": 1.42, "y": 0.83, "z": 0.91}


def test_located_in_with_full_pose() -> None:
    record = {
        "from": "item:mug",
        "to": "space:kitchen",
        "position": {"x": 1.42, "y": 0.83, "z": 0.91},
        "orientation": {"yaw": 0.0, "pitch": 0.0, "roll": 0.0},
    }
    apply_located_in_pose(record, target_type="space", source="located_in.json")
    assert record["orientation"] == {"pitch": 0.0, "roll": 0.0, "yaw": 0.0}


def test_space_may_be_located_in_another_space() -> None:
    assert located_in_endpoints_allowed("space", "space")
    assert located_in_endpoints_allowed("item", "space")
    assert located_in_endpoints_allowed("item", "address")
    assert not located_in_endpoints_allowed("space", "item")
    record = {
        "from": "space:refrigerator",
        "to": "space:kitchen",
        "position": {"x": 3.1, "y": 0.4, "z": 0.0},
        "orientation": {"yaw": 1.57, "pitch": 0.0, "roll": 0.0},
    }
    apply_located_in_pose(record, target_type="space", source="located_in.json")
    assert record["position"]["x"] == 3.1


@pytest.mark.parametrize(
    "mutate, match",
    [
        (lambda r: r["coordinate"].update(origin=[1, math.inf, 0]), "finite"),
        (lambda r: r["coordinate"].update(origin=[1, math.nan, 0]), "finite"),
        (lambda r: r.update(navigable=1), "boolean"),
        (lambda r: r["geometry"].update(size=[4.8, 3.6, 0]), "positive"),
        (lambda r: r["geometry"].update(type="mesh"), "box"),
        (lambda r: r["coordinate"].update(unit="inch"), "must be 'm'"),
        (lambda r: r["coordinate"].update(z_axis=[0, 0, 0]), "zero vector"),
    ],
)
def test_invalid_space_spatial_values(mutate, match) -> None:
    record = _space(
        coordinate={"unit": "m", "origin": [0.0, 0.0, 0.0], "z_axis": "up"},
        geometry={"type": "box", "size": [4.8, 3.6, 2.7]},
        navigable=True,
        accessible=True,
    )
    mutate(record)
    with pytest.raises(SpatialContractError, match=match):
        apply_space_spatial_fields(record, source="space.json")


def test_pose_requires_space_target() -> None:
    record = {
        "from": "item:house",
        "to": "address:home",
        "position": {"x": 0.0, "y": 0.0, "z": 0.0},
    }
    with pytest.raises(SpatialContractError, match="space target"):
        apply_located_in_pose(record, target_type="address", source="located_in.json")


def test_invalid_located_in_target_type() -> None:
    registry = EdgeSchemaRegistry.load_default()
    with pytest.raises(ValueError, match="Invalid located_in endpoints"):
        registry.validate_endpoints("located_in", "item", "person")
    assert not located_in_endpoints_allowed("space", "address")
    assert not located_in_endpoints_allowed("person", "space")


def test_spatial_fields_are_rejected_on_non_space_nodes() -> None:
    with pytest.raises(SpatialContractError, match="belongs to space"):
        reject_non_space_spatial_fields(
            {"id": "item:mug", "navigable": True},
            "item",
            source="item.json",
        )


def test_existing_household_records_remain_valid() -> None:
    for record in json.loads((PRODUCTION_DATA / "nodes" / "space.json").read_text()):
        apply_space_spatial_fields(record, source="data/nodes/space.json")
    for record in json.loads((PRODUCTION_DATA / "edges" / "located_in.json").read_text()):
        target_type = record["to"].partition(":")[0]
        apply_located_in_pose(
            record, target_type=target_type, source="data/edges/located_in.json"
        )


@pytest.mark.asyncio
async def test_ingestion_export_round_trip_preserves_spatial_data(tmp_path: Path) -> None:
    data_dir = tmp_path / "graph"
    copytree(STATIC_TEST_DATA, data_dir)
    spaces = json.loads((data_dir / "nodes" / "space.json").read_text())
    kitchen = next(space for space in spaces if space["id"] == "space:kitchen")
    kitchen.update(
        {
            "coordinate": {
                "unit": "m",
                "origin": [0.0, 0.0, 0.0],
                "z_axis": "up",
            },
            "geometry": {"type": "box", "size": [4.8, 3.6, 2.7]},
            "navigable": True,
            "accessible": True,
        }
    )
    (data_dir / "nodes" / "space.json").write_text(json.dumps(spaces))
    edges = json.loads((data_dir / "edges" / "located_in.json").read_text())
    milk = next(edge for edge in edges if edge["from"] == "item:milk")
    milk["position"] = {"x": 0.2, "y": 0.1, "z": 0.4}
    milk["orientation"] = {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}
    edges.append(
        {
            "from": "space:test_house:kitchen:fridge_01:interior",
            "to": "space:kitchen",
            "position": {"x": 1.0, "y": 0.5, "z": 0.0},
        }
    )
    (data_dir / "edges" / "located_in.json").write_text(json.dumps(edges))

    database = MemoryDatabase()
    await database.connect()
    try:
        await ingest_directory(database, data_dir)  # type: ignore[arg-type]
        target = tmp_path / "export"
        await export_directory(database, target)  # type: ignore[arg-type]
    finally:
        await database.close()

    exported_spaces = {
        row["id"]: row
        for row in json.loads((target / "nodes" / "space.json").read_text())
    }
    assert exported_spaces["space:kitchen"]["coordinate"]["unit"] == "m"
    assert exported_spaces["space:kitchen"]["geometry"]["type"] == "box"
    assert exported_spaces["space:kitchen"]["navigable"] is True
    assert exported_spaces["space:kitchen"]["accessible"] is True
    exported_edges = json.loads((target / "edges" / "located_in.json").read_text())
    by_from = {edge["from"]: edge for edge in exported_edges}
    assert by_from["item:milk"]["position"] == {"x": 0.2, "y": 0.1, "z": 0.4}
    assert by_from["item:milk"]["orientation"] == {
        "pitch": 0.0,
        "roll": 0.0,
        "yaw": 0.0,
    }
    assert by_from["item:test_house"] == {
        "from": "item:test_house",
        "to": "address:test_house",
    }
    assert by_from["space:test_house:kitchen:fridge_01:interior"]["to"] == "space:kitchen"
    assert by_from["space:test_house:kitchen:fridge_01:interior"]["position"] == {
        "x": 1.0,
        "y": 0.5,
        "z": 0.0,
    }
