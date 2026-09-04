from pathlib import Path

import pytest
import yaml

from home_cortex.edge_schema import EdgeSchemaRegistry
from home_cortex.schema_catalog import RuntimeSchemaCatalog
from home_cortex.semantic_facts import SemanticSchemaRegistry, TierZeroSemanticParser
from home_cortex.semantic_ontology import SemanticOntology

ROOT = Path(__file__).parents[1]
ONTOLOGY_PATH = ROOT / "schemas" / "semantic" / "ontology.yaml"


def test_default_ontology_owns_property_and_kinship_semantics() -> None:
    ontology = SemanticOntology.from_file(ONTOLOGY_PATH)

    assert ontology.property_fields("birth_date") == (
        "birth_date",
        "birthday",
        "dob",
        "date_of_birth",
    )
    father_in_law = ontology.reference_concepts["father_in_law"]
    assert [step.relation for step in father_in_law.path] == ["spouse", "parent"]
    assert father_in_law.path[-1].filters[0].value == "male"
    assert [
        step.filters[0].value
        for step in ontology.reference_concepts["paternal_grandson"].path
    ] == ["male", "male"]
    assert [
        step.filters[0].value
        for step in ontology.reference_concepts["maternal_grandson"].path
    ] == ["female", "male"]
    older_brother = ontology.reference_concepts["older_brother"]
    age_filter = older_brother.path[-1].filters[-1]
    assert age_filter.operator == "lt"
    assert age_filter.value_from == "anchor"


def test_new_kinship_alias_requires_only_ontology_change(tmp_path: Path) -> None:
    raw = yaml.safe_load(ONTOLOGY_PATH.read_text(encoding="utf-8"))
    raw["reference_concepts"]["father_in_law"]["aliases"].append("老丈人")
    custom_path = tmp_path / "ontology.yaml"
    custom_path.write_text(
        yaml.safe_dump(raw, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    request = TierZeroSemanticParser(
        SemanticOntology.from_file(custom_path)
    ).parse("我老丈人是谁")

    assert request is not None
    assert [step.relation for step in request.subject.path] == ["spouse", "parent"]


def test_relation_direction_is_derived_from_edge_schema_metadata() -> None:
    edge_registry = EdgeSchemaRegistry.from_directory(ROOT / "schemas" / "edge")
    catalog = RuntimeSchemaCatalog.from_data_dir(ROOT / "data", edge_registry)
    schema = SemanticSchemaRegistry(catalog)

    assert schema.physical_relation("spouse") == ("spouse_of", None)
    assert schema.physical_relation("child") == ("parent_of", "out")
    assert schema.physical_relation("parent") == ("parent_of", "in")
    assert schema.physical_relation("member") == ("lives_in", "in")
    assert schema.physical_relation("residence") == ("lives_in", "out")


def test_ontology_rejects_unknown_base_relation(tmp_path: Path) -> None:
    raw = yaml.safe_load(ONTOLOGY_PATH.read_text(encoding="utf-8"))
    raw["reference_concepts"]["son"]["path"][0]["relation"] = "invented"
    custom_path = tmp_path / "ontology.yaml"
    custom_path.write_text(
        yaml.safe_dump(raw, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Unknown base relation"):
        SemanticOntology.from_file(custom_path)
