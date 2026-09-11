"""Model-facing item mutation intents; storage identity is deliberately absent."""
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, RootModel, TypeAdapter, model_validator

from .writing import WriteMode


class _NamedWrite(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)
    item_name: str = Field(min_length=1, max_length=256, description=(
        "Only the item's literal noun phrase. Exclude the instruction verb, "
        "quantity, and surrounding location/existence clause."
    ))
    mode: WriteMode = "commit"


class NamedCreateItem(_NamedWrite):
    operation: Literal["create"]
    location_name: str = Field(min_length=1, max_length=256, description=(
        "Complete literal name of the destination, including subspace qualifiers."
    ))


class NamedMoveItem(_NamedWrite):
    operation: Literal["update_location"]
    location_name: str = Field(min_length=1, max_length=256, description=(
        "Complete literal name of the destination, including subspace qualifiers."
    ))


class NamedDeleteItem(_NamedWrite):
    operation: Literal["delete"]


NamedWriteRequest = Annotated[
    NamedCreateItem | NamedMoveItem | NamedDeleteItem,
    Field(discriminator="operation"),
]
NAMED_WRITE_ADAPTER = TypeAdapter(NamedWriteRequest)


class NamedWriteItemArguments(RootModel[NamedWriteRequest]):
    model_config = ConfigDict(strict=True)



class MutationDecision(BaseModel):
    model_config = ConfigDict(extra='forbid')
    requires_mutation: bool
    mutation: NamedCreateItem | NamedMoveItem | NamedDeleteItem | None = None

    @model_validator(mode='after')
    def validate_intent(self):
        if not self.requires_mutation and self.mutation is not None:
            raise ValueError('A non-mutation decision cannot carry a write')
        return self


def mutation_messages(messages):
    """Interpret the current speech act without household facts or storage fields."""
    import json
    latest = next(str(message.get('content', '')) for message in reversed(messages)
                  if message.get('role') == 'user')
    instructions = (
        'Classify the latest user request and compile at most one item-state change. '
        'Explicit requests to record/remember an item at a location use create; '
        'moving an existing item uses update_location; explicit record removal uses delete. '
        'Copy the item noun phrase into item_name and the complete destination noun phrase '
        'into location_name. For a report that place Y contains X, the item is X, not Y '
        'and not the entire clause. Preserve qualifiers inside names. Use mode=commit '
        'unless preview is requested. Do not invent names or IDs. '
        'Questions about existing facts, translations, quotes, hypotheticals, and negated '
        'writes require requires_mutation=false and mutation=null. An explicit write '
        'requires requires_mutation=true; if the needed names are missing, mutation=null '
        'so the assistant can ask for clarification. Return JSON only.'
    )
    examples = [
        ('请记下：茶几下层放着画册。', {'requires_mutation': True, 'mutation': {
            'operation': 'create', 'item_name': '画册', 'location_name': '茶几下层', 'mode': 'commit'}}),
        ('Move the sketchbook to the studio shelf.', {'requires_mutation': True, 'mutation': {
            'operation': 'update_location', 'item_name': 'sketchbook', 'location_name': 'studio shelf', 'mode': 'commit'}}),
        ('茶几下层放着什么？', {'requires_mutation': False, 'mutation': None}),
        ('Translate: "Record a notebook in the desk drawer."', {'requires_mutation': False, 'mutation': None}),
    ]
    result = [{'role': 'system', 'content': instructions}]
    for utterance, decision in examples:
        result.extend([{'role': 'user', 'content': utterance},
                       {'role': 'assistant', 'content': json.dumps(decision, ensure_ascii=False)}])
    result.append({'role': 'user', 'content': latest})
    return result


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
