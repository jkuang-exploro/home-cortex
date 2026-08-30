import json
import re
from collections.abc import Mapping, Sequence
from typing import Any, Literal

from .text import latest_user_message, normalize_language_code

INTERNAL_ID_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_])"
    r"[A-Za-z_][A-Za-z0-9_]*(?::[A-Za-z0-9_-]+)+"
)
_TRAILING_ID_CHARACTERS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_:-"
)
_CHINESE_PATTERN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
_INTERNAL_ID_REQUEST_PATTERNS = (
    re.compile(
        r"\b(?:internal|database|record|object|graph)\s+(?:id|identifier)s?\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bdebug(?:ging)?\b", re.IGNORECASE),
    re.compile(r"(?:内部|数据库|记录|对象|图)(?:的)?\s*(?:ID|id|标识符)"),
    re.compile(r"调试|技术细节"),
)
ReferenceMode = Literal["address", "name", "id"]


class DisplayNameResolver:
    """Resolve stable graph IDs to language-appropriate display names."""

    def __init__(self, values: Sequence[Any] = ()) -> None:
        self._entities: dict[str, Mapping[str, Any]] = {}
        for value in values:
            self.register(value)

    @classmethod
    def from_messages(
        cls,
        messages: Sequence[Mapping[str, Any]],
        values: Sequence[Any] = (),
    ) -> "DisplayNameResolver":
        resolver = cls(values)
        for message in messages:
            if message.get("role") != "tool":
                continue
            content = message.get("content")
            if not isinstance(content, str):
                continue
            try:
                resolver.register(json.loads(content))
            except (json.JSONDecodeError, TypeError):
                continue
        return resolver

    def register(self, value: Any) -> None:
        """Recursively index entity metadata without altering tool results."""
        if isinstance(value, Mapping):
            record_id = value.get("id")
            if isinstance(record_id, str) and INTERNAL_ID_PATTERN.fullmatch(record_id):
                existing = self._entities.get(record_id)
                self._entities[record_id] = (
                    {**value, **existing}
                    if isinstance(existing, Mapping)
                    else value
                )
            for item in value.values():
                self.register(item)
        elif isinstance(value, Sequence) and not isinstance(
            value,
            (str, bytes, bytearray),
        ):
            for item in value:
                self.register(item)

    def resolve(
        self,
        object_or_id: Any,
        language: str = "en",
        *,
        mode: ReferenceMode = "name",
    ) -> str:
        if isinstance(object_or_id, Mapping):
            entity = object_or_id
            record_id = entity.get("id")
        else:
            record_id = str(object_or_id)
            entity = self._entities.get(record_id)

        if mode == "id":
            return str(record_id)
        if isinstance(entity, Mapping):
            if mode == "address" and str(record_id).startswith("person:"):
                address_as = _select_address_as(entity, language)
                if address_as:
                    return address_as
            display_name = _select_display_name(entity, language)
            if display_name:
                return display_name
        return str(record_id)

    def render(
        self,
        text: str,
        language: str = "en",
        *,
        expose_internal_ids: bool = False,
        person_mode: ReferenceMode = "address",
    ) -> str:
        if expose_internal_ids:
            return text

        def replace(match: re.Match[str]) -> str:
            record_id = match.group(0)
            mode = person_mode if record_id.startswith("person:") else "name"
            return self.resolve(record_id, language, mode=mode)

        return INTERNAL_ID_PATTERN.sub(replace, text)


class DisplayTextStream:
    """Incrementally render streamed text without leaking split record IDs."""

    def __init__(
        self,
        resolver: DisplayNameResolver,
        language: str,
        *,
        expose_internal_ids: bool = False,
        person_mode: ReferenceMode = "address",
    ) -> None:
        self.resolver = resolver
        self.language = language
        self.expose_internal_ids = expose_internal_ids
        self.person_mode = person_mode
        self._pending = ""

    def feed(self, text: str) -> str:
        if self.expose_internal_ids:
            return text
        combined = self._pending + text
        boundary = len(combined)
        while boundary and combined[boundary - 1] in _TRAILING_ID_CHARACTERS:
            boundary -= 1
        ready = combined[:boundary]
        self._pending = combined[boundary:]
        return self.resolver.render(
            ready,
            self.language,
            person_mode=self.person_mode,
        )

    def finish(self) -> str:
        rendered = self.resolver.render(
            self._pending,
            self.language,
            person_mode=self.person_mode,
        )
        self._pending = ""
        return rendered


def resolve_display_name(object_or_id: Any, language: str = "en") -> str:
    """Resolve a record object directly; bare unknown IDs safely remain IDs."""
    return DisplayNameResolver().resolve(object_or_id, language)


def resolve_person_reference(
    person_or_id: Any,
    language: str = "en",
    *,
    mode: ReferenceMode = "address",
    resolver: DisplayNameResolver | None = None,
) -> str:
    """Resolve a person explicitly as an address, name, or stable ID."""
    if resolver is None:
        values = [person_or_id] if isinstance(person_or_id, Mapping) else []
        resolver = DisplayNameResolver(values)
    return resolver.resolve(person_or_id, language, mode=mode)


def conversation_language(messages: Sequence[Mapping[str, Any]]) -> str:
    content = latest_user_message(messages)
    lowered = content.casefold()
    if (
        "in english" in lowered
        or "answer in english" in lowered
        or "用英文" in content
    ):
        return "en"
    if (
        "in chinese" in lowered
        or "answer in chinese" in lowered
        or "用中文" in content
    ):
        return "zh"
    return "zh" if _CHINESE_PATTERN.search(content) else "en"


def internal_ids_requested(messages: Sequence[Mapping[str, Any]]) -> bool:
    content = latest_user_message(messages)
    return any(pattern.search(content) for pattern in _INTERNAL_ID_REQUEST_PATTERNS)


def _select_display_name(entity: Mapping[str, Any], language: str) -> str | None:
    language = normalize_language_code(language)
    name = entity.get("name")
    display_name = entity.get("display_name")

    localized = _localized_value(name, language) or _localized_value(
        display_name,
        language,
    )
    if localized:
        return localized

    for default in (
        entity.get("default_display_name"),
        _default_value(display_name),
        _default_value(name),
    ):
        if isinstance(default, str) and default.strip():
            return default.strip()
    return None


def _select_address_as(entity: Mapping[str, Any], language: str) -> str | None:
    language = normalize_language_code(language)
    address_as = entity.get("address_as")
    localized = _localized_value(address_as, language)
    if localized:
        return localized

    for fallback in (
        entity.get("default_address_as"),
        _explicit_default_value(address_as),
        _another_value(address_as),
    ):
        if isinstance(fallback, str) and fallback.strip():
            return fallback.strip()
    return None


def _localized_value(value: Any, language: str) -> str | None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if normalize_language_code(str(key)) == language:
                if isinstance(item, str) and item.strip():
                    return item.strip()
        return None
    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        for item in value:
            if not isinstance(item, str) or not item.strip():
                continue
            is_chinese = _CHINESE_PATTERN.search(item) is not None
            if (language == "zh" and is_chinese) or (
                language != "zh" and not is_chinese
            ):
                return item.strip()
    return None


def _default_value(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        default = value.get("default")
        if isinstance(default, str):
            return default
        return next(
            (item for item in value.values() if isinstance(item, str) and item.strip()),
            None,
        )
    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        return next(
            (item for item in value if isinstance(item, str) and item.strip()),
            None,
        )
    return None


def _explicit_default_value(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        default = value.get("default")
        return default if isinstance(default, str) else None
    return None


def _another_value(value: Any) -> str | None:
    if isinstance(value, Mapping):
        return next(
            (
                item
                for key, item in value.items()
                if str(key).casefold() != "default"
                and isinstance(item, str)
                and item.strip()
            ),
            None,
        )
    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        return next(
            (item for item in value if isinstance(item, str) and item.strip()),
            None,
        )
    return None



