from pathlib import Path

import pytest
import yaml

from home_cortex.edge_schema import EdgeSchemaRegistry
from home_cortex.schema_catalog import RuntimeSchemaCatalog
from home_cortex.semantic_facts import SemanticSchemaRegistry
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
    assert ontology.property_fields("item_type") == ("item_type",)
    payload = ontology.planner_payload()["properties"]
    assert "类别" in payload["item_type"]


def test_new_kinship_alias_is_exposed_to_planner_by_ontology_change(
    tmp_path: Path,
) -> None:
    raw = yaml.safe_load(ONTOLOGY_PATH.read_text(encoding="utf-8"))
    raw["reference_concepts"]["father_in_law"]["aliases"].append("老丈人")
    custom_path = tmp_path / "ontology.yaml"
    custom_path.write_text(
        yaml.safe_dump(raw, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    payload = SemanticOntology.from_file(custom_path).planner_payload()
    concept = payload["reference_concepts"]["father_in_law"]

    assert "老丈人" in concept["aliases"]
    assert [step["relation"] for step in concept["path"]] == ["spouse", "parent"]


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


def test_display_metadata_does_not_change_planner_contract(tmp_path: Path) -> None:
    raw = yaml.safe_load(ONTOLOGY_PATH.read_text())
    labeled = SemanticOntology.from_file(ONTOLOGY_PATH)
    for group in ('properties', 'collection_predicates', 'reference_concepts'):
        for definition in raw[group].values():
            definition.pop('label', None)
            definition.pop('value_labels', None)
    path = tmp_path / 'legacy.yaml'
    path.write_text(yaml.safe_dump(raw))
    legacy = SemanticOntology.from_file(path)
    assert legacy.planner_payload() == labeled.planner_payload()
    assert labeled.properties['gender'].label == (('en', 'gender'), ('zh', '性别'))
    assert legacy.properties['gender'].label == ()
    catalog = RuntimeSchemaCatalog.from_data_dir(tmp_path, EdgeSchemaRegistry.load_default())
    before, after = (SemanticSchemaRegistry(catalog, ontology) for ontology in (legacy, labeled))
    assert before.planner_capability_payload() == after.planner_capability_payload()
    assert before.planner_output_schema() == after.planner_output_schema()


@pytest.mark.parametrize('field,value', [
    ('label', {'en': ''}), ('label', {'en': 42}), ('label', ['gender']),
    ('value_labels', {'female': {'en': False}}), ('value_labels', {'female': 'female'}),
])
def test_display_metadata_rejects_malformed_labels(tmp_path: Path, field, value) -> None:
    raw = yaml.safe_load(ONTOLOGY_PATH.read_text())
    raw['properties']['gender'][field] = value
    path = tmp_path / 'invalid.yaml'
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError):
        SemanticOntology.from_file(path)
