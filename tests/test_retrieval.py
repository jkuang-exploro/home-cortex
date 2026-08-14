import pytest
from surrealdb import RecordID

from home_cortex.retrieval import RetrievalService, to_json_value


class FakeDatabase:
    async def select(self, table: str):
        return [{"id": f"{table}:one", "name": table}]


@pytest.mark.asyncio
async def test_retrieval_formats_graph_context() -> None:
    service = RetrievalService(FakeDatabase())  # type: ignore[arg-type]

    result = await service.retrieve("Who lives here?")

    assert result.nodes["person"][0]["name"] == "person"
    assert result.edges["resides_in"][0]["name"] == "resides_in"
    assert '"nodes"' in result.text


def test_record_id_is_serialized() -> None:
    assert to_json_value(RecordID("person", "alice")) == "person:alice"
