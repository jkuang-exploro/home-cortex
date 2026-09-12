"""Model-facing item mutation intents; physical table IDs are absent."""
import json
from functools import lru_cache
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, RootModel, TypeAdapter, model_validator

from .text import latest_user_message
WriteMode = Literal["preview", "commit"]


def _compact_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


class _NamedWrite(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)
    item_name: str = Field(min_length=1, max_length=256, description=(
        "Only the item's literal noun phrase. Exclude the instruction verb, "
        "quantity, and surrounding location/existence clause."
    ))
    mode: WriteMode = "commit"


AttributeValue = str | int | float | bool


class NamedUpdateAttributes(_NamedWrite):
    operation: Literal["update_attributes"]
    attributes: dict[str, AttributeValue] = Field(min_length=1, max_length=32,
        description="Only explicitly requested semantic attribute changes; preserve all other fields.")


class NamedCreateItem(_NamedWrite):
    attributes: dict[str, AttributeValue] = Field(default_factory=dict, max_length=32,
        description="Semantic item attributes, including item_type. Unknown category is unknown; do not invent other values.")
    operation: Literal["create"]
    location_name: str = Field(min_length=1, max_length=256, description=(
        "Complete literal name of the destination, including subspace qualifiers."
    ))
    name_en: str = Field(min_length=1, max_length=256, description=(
        "English display name. Copy the item noun when the user spoke English; "
        "translate it into English when the user spoke Chinese."
    ))
    name_zh: str = Field(min_length=1, max_length=256, description=(
        "Chinese display name. Copy the item noun when the user spoke Chinese; "
        "translate it into Chinese when the user spoke English."
    ))
    item_key: str = Field(
        pattern=r"^[A-Za-z][A-Za-z0-9_-]{0,63}$",
        description=(
            "Readable ASCII record key, without an item: prefix. Use pinyin for "
            "Chinese names (玉米→yumi) and lowercase_underscore English for "
            "English names (red screwdriver→red_screwdriver). Never a hash."
        ),
    )

    @model_validator(mode="after")
    def reject_hashed_item_key(self) -> "NamedCreateItem":
        key = self.item_key
        if key.startswith("recorded_") and len(key) > 20:
            raise ValueError("item_key must be a readable name encoding, not a hash")
        return self


class NamedMoveItem(_NamedWrite):
    operation: Literal["update_location"]
    location_name: str = Field(min_length=1, max_length=256, description=(
        "Complete literal name of the destination, including subspace qualifiers."
    ))


class NamedDeleteItem(_NamedWrite):
    operation: Literal["delete"]


NamedWriteRequest = Annotated[
    NamedCreateItem | NamedMoveItem | NamedUpdateAttributes | NamedDeleteItem,
    Field(discriminator="operation"),
]
NAMED_WRITE_ADAPTER = TypeAdapter(NamedWriteRequest)


class NamedWriteItemArguments(RootModel[NamedWriteRequest]):
    model_config = ConfigDict(strict=True)



class MutationDecision(BaseModel):
    model_config = ConfigDict(extra='forbid')
    requires_mutation: bool
    mutation: NamedCreateItem | NamedMoveItem | NamedUpdateAttributes | NamedDeleteItem | None = None

    @model_validator(mode='after')
    def validate_intent(self):
        if not self.requires_mutation and self.mutation is not None:
            raise ValueError('A non-mutation decision cannot carry a write')
        return self


@lru_cache(maxsize=1)
def _mutation_prefix() -> tuple[tuple[str, str], ...]:
    """Static mutation compiler prefix; ontology load is cached with the text."""
    from .semantic_ontology import SemanticOntology
    attributes = {name: {"type": prop.item_writable, "aliases": prop.aliases,
                         "value_labels": dict(prop.value_labels), "write_hint": prop.write_hint}
                  for name, prop in SemanticOntology.load_default().properties.items()
                  if prop.item_writable}
    instructions = (
        'Classify the latest user request and compile at most one item-state change. '
        'Explicit requests to record/remember an item at a location use create; '
        'moving an existing item uses update_location; changing its attributes uses update_attributes; explicit record removal uses delete. '
        'Copy the item noun phrase into item_name and the complete destination noun phrase '
        'into location_name. For a report that place Y contains X, the item is X, not Y '
        'and not the entire clause. Preserve qualifiers inside names. Use mode=commit '
        'unless preview is requested. Do not invent household graph IDs. '
        'For create, always emit name_en, name_zh, and item_key. Display names must include '
        'both English and Chinese: if the user named the item in Chinese, copy that into '
        'name_zh and translate into English for name_en; if the user named it in English, '
        'copy that into name_en and translate into Chinese for name_zh. Never use language '
        'tag und or a single-language name object. item_key is a readable ASCII encoding of '
        'the item (pinyin for Chinese, lowercase_underscore English for English), not a hash '
        'and not prefixed with item:. '
        'For create, include attributes.item_type: use an explicitly stated category, or '
        'infer an unambiguous common category from the item noun; otherwise use unknown. '
        'Use canonical category values when declared; do not infer brand, quantity, color, '
        'model, dates, or other attributes. Copy other attributes only when explicitly given. '
        'description is only for an explicitly supplied description, never the command or a generated summary. '
        'Keep identifying adjectives in the complete item_name while also recording explicitly stated attributes. '
        'An attribute update needs item_name and attributes, not location_name. '
        'Only include fields the user asked to change. Do not silently replace existing '
        'attributes during create. Attribute names and types are declared below. '
        'Questions asking for attributes are reads, not updates. '

        'Questions about existing facts, translations, quotes, hypotheticals, and negated '
        'writes require requires_mutation=false and mutation=null. An explicit write '
        'requires requires_mutation=true; if the needed names are missing, mutation=null '
        'so the assistant can ask for clarification. Return JSON only.\n'
        + _compact_json(attributes)
    )
    examples = [
        ('请记下：茶几下层放着画册。', {'requires_mutation': True, 'mutation': {
            'operation': 'create', 'item_name': '画册', 'location_name': '茶几下层',
            'name_en': 'picture album', 'name_zh': '画册', 'item_key': 'huace',
            'attributes': {'item_type': 'book'}, 'mode': 'commit'}}),
        ('Move the sketchbook to the studio shelf.', {'requires_mutation': True, 'mutation': {
            'operation': 'update_location', 'item_name': 'sketchbook', 'location_name': 'studio shelf', 'mode': 'commit'}}),
        ('把台灯的颜色改为蓝色。', {'requires_mutation': True, 'mutation': {
            'operation': 'update_attributes', 'item_name': '台灯', 'attributes': {'color': '蓝色'}, 'mode': 'commit'}}),
        ('Record a red screwdriver in the workshop drawer.', {'requires_mutation': True, 'mutation': {
            'operation': 'create', 'item_name': 'red screwdriver', 'location_name': 'workshop drawer',
            'name_en': 'red screwdriver', 'name_zh': '红色螺丝刀', 'item_key': 'red_screwdriver',
            'attributes': {'item_type': 'tool', 'color': 'red'}, 'mode': 'commit'}}),
        ('Record Zeta-17 in the studio tray.', {'requires_mutation': True, 'mutation': {
            'operation': 'create', 'item_name': 'Zeta-17', 'location_name': 'studio tray',
            'name_en': 'Zeta-17', 'name_zh': 'Zeta-17', 'item_key': 'zeta_17',
            'attributes': {'item_type': 'unknown'}, 'mode': 'commit'}}),
        ('茶几下层放着什么？', {'requires_mutation': False, 'mutation': None}),
        ('Translate: "Record a notebook in the desk drawer."', {'requires_mutation': False, 'mutation': None}),
    ]
    messages = [("system", instructions)]
    for utterance, decision in examples:
        messages.extend((("user", utterance), ("assistant", _compact_json(decision))))
    return tuple(messages)


def mutation_messages(messages):
    """Interpret the current speech act without household facts or storage fields."""
    return [
        {"role": role, "content": content}
        for role, content in _mutation_prefix()
    ] + [{"role": "user", "content": latest_user_message(messages)}]


def read_plan_schema(schema):
    """Keep the read compiler's schema stable as the overall intent union grows."""
    from copy import deepcopy
    result = deepcopy(dict(schema))
    if 'mutation' not in result.get('properties', {}):
        return result
    result['properties'].pop('mutation', None)
    definitions = result.get('$defs', {})
    needed = set()

    def visit(node):
        if isinstance(node, dict):
            reference = node.get('$ref', '')
            if reference.startswith('#/$defs/'):
                name = reference.rsplit('/', 1)[-1]
                if name not in needed:
                    needed.add(name)
                    visit(definitions[name])
            for key, child in node.items():
                if key != '$defs':
                    visit(child)
        elif isinstance(node, list):
            for child in node:
                visit(child)
    visit(result)
    result['$defs'] = {name: value for name, value in definitions.items() if name in needed}
    return result


def attribute_output_schema(schema):
    """Constrain generated attribute keys/types using the declared semantic vocabulary."""
    from copy import deepcopy
    from .semantic_ontology import SemanticOntology
    result = deepcopy(schema)
    properties = {}
    for name, prop in SemanticOntology.load_default().properties.items():
        if prop.item_writable:
            properties[name] = {'type': 'string' if prop.item_writable == 'date' else prop.item_writable}
            if prop.write_hint:
                properties[name]['description'] = prop.write_hint
            if prop.item_writable == 'date':
                properties[name]['format'] = 'date'
    for definition in result.get('$defs', {}).values():
        attributes = definition.get('properties', {}).get('attributes')
        if attributes is not None:
            attributes['additionalProperties'] = False
            attributes['properties'] = properties
    return result
