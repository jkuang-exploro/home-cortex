from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from .agents import AgentDefinition
from .display import resolve_display_name, resolve_person_reference

ReceptionCategory = Literal[
    "owner",
    "minor_dependent",
    "adult_dependent",
    "guest",
    "unknown",
]

RECEPTION_CATEGORIES = frozenset(
    {"owner", "minor_dependent", "adult_dependent", "guest", "unknown"}
)


class RelationshipRetrieval(Protocol):
    async def search_entities(
        self,
        text: str,
        entity_type: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]: ...

    async def get_relationships(
        self,
        entity_id: str,
        relation: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]: ...


@dataclass(frozen=True)
class Greeting:
    text: str
    language: str
    reception_category: ReceptionCategory


class GreetingResolver:
    """Render an agent's deterministic reception policy from trusted graph state."""

    def __init__(self, settings: Mapping[str, Any]) -> None:
        reception = settings.get("reception")
        if not isinstance(reception, Mapping):
            raise ValueError("Agent settings require a reception policy")
        self.default_language = _language_code(
            _required_string(reception, "default_language")
        )
        self.templates = _required_mapping(reception, "greetings")
        overrides = reception.get("person_overrides", {})
        if not isinstance(overrides, Mapping):
            raise ValueError("reception.person_overrides must be an object")
        self.person_overrides = overrides

    def resolve(
        self,
        *,
        person: Mapping[str, Any] | None,
        household: Mapping[str, Any] | None,
        relationships: Sequence[Mapping[str, Any]],
        household_id: str,
        language: str,
    ) -> Greeting:
        language_code = _language_code(language)
        person_id = _record_id(person)
        category = _reception_category(
            relationships,
            person_id=person_id,
            household_id=household_id,
        )
        address_as = _human_person_reference(person, language_code)
        household_name = _human_household_name(household, language_code)
        template_category = category
        if category != "unknown" and not address_as:
            template_category = "unknown"
        if category == "guest" and not household_name:
            template_category = "unknown"
        template = self._template(
            language_code,
            template_category,
            person_id=person_id,
        )
        rendered = template.format(
            address_as=address_as,
            household_name=household_name,
        ).strip()
        if not rendered or "person:" in rendered or "location:" in rendered:
            raise ValueError("Greeting policy produced an unsafe greeting")
        return Greeting(
            text=rendered,
            language=language_code,
            reception_category=category,
        )

    def _template(
        self,
        language: str,
        category: ReceptionCategory,
        *,
        person_id: str | None,
    ) -> str:
        for selected_language in _language_fallbacks(
            language,
            self.default_language,
        ):
            if person_id is not None and category != "unknown":
                override = _nested_string(
                    self.person_overrides,
                    selected_language,
                    category,
                    person_id,
                )
                if override:
                    return override
            category_template = _nested_string(
                self.templates,
                selected_language,
                category,
            )
            if category_template:
                return category_template
            generic_template = _nested_string(
                self.templates,
                selected_language,
                "generic",
            )
            if generic_template:
                return generic_template
        raise ValueError(
            f"No greeting template is configured for category {category!r}"
        )


class GreetingService:
    """Load trusted graph context, then delegate rendering to a pure resolver."""

    def __init__(self, retrieval: RelationshipRetrieval) -> None:
        self.retrieval = retrieval

    async def resolve(
        self,
        definition: AgentDefinition,
        person: Mapping[str, Any] | None,
        language: str,
    ) -> Greeting:
        household_id = definition.settings.get("home_entity_id")
        if not isinstance(household_id, str) or not household_id:
            raise ValueError(
                f"Agent {definition.id!r} requires settings.home_entity_id"
            )

        household_records = await self.retrieval.search_entities(
            household_id,
            entity_type="location",
            limit=1,
        )
        household = next(
            (
                record
                for record in household_records
                if record.get("id") == household_id
            ),
            None,
        )
        person_id = _record_id(person)
        relationships = (
            await self.retrieval.get_relationships(person_id)
            if person_id is not None
            else []
        )
        return GreetingResolver(definition.settings).resolve(
            person=person,
            household=household,
            relationships=relationships,
            household_id=household_id,
            language=language,
        )


def _reception_category(
    relationships: Sequence[Mapping[str, Any]],
    *,
    person_id: str | None,
    household_id: str,
) -> ReceptionCategory:
    if person_id is None:
        return "unknown"
    roles: set[str] = set()
    for relationship in relationships:
        endpoints = {relationship.get("in"), relationship.get("out")}
        related = relationship.get("related_entity")
        if isinstance(related, Mapping):
            endpoints.add(related.get("id"))
        if person_id not in endpoints or household_id not in endpoints:
            continue
        role = relationship.get("household_role")
        if isinstance(role, str) and role in RECEPTION_CATEGORIES - {"unknown"}:
            roles.add(role)
    if len(roles) != 1:
        return "unknown"
    return next(iter(roles))  # type: ignore[return-value]


def _human_person_reference(
    person: Mapping[str, Any] | None,
    language: str,
) -> str:
    if person is None:
        return ""
    reference = resolve_person_reference(person, language, mode="address")
    if reference == _record_id(person):
        return ""
    return reference


def _human_household_name(
    household: Mapping[str, Any] | None,
    language: str,
) -> str:
    if household is None:
        return ""
    name = resolve_display_name(household, language)
    if name == _record_id(household):
        return ""
    return name


def _record_id(value: Mapping[str, Any] | None) -> str | None:
    if value is None:
        return None
    record_id = value.get("id")
    return record_id if isinstance(record_id, str) else None


def _language_code(language: str) -> str:
    if not isinstance(language, str) or not language.strip():
        raise ValueError("language must be a non-empty string")
    return language.casefold().split("-", 1)[0]


def _language_fallbacks(language: str, default: str) -> tuple[str, ...]:
    return (language,) if language == default else (language, default)


def _required_mapping(value: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    item = value.get(key)
    if not isinstance(item, Mapping):
        raise ValueError(f"reception.{key} must be an object")
    return item


def _required_string(value: Mapping[str, Any], key: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item.strip():
        raise ValueError(f"reception.{key} must be a non-empty string")
    return item.strip()


def _nested_string(value: Mapping[str, Any], *keys: str) -> str | None:
    current: Any = value
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current.strip() if isinstance(current, str) and current.strip() else None
