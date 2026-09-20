"""Shared text helpers for conversation input, language codes, and logs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def latest_user_message(messages: Sequence[Mapping[str, Any]]) -> str:
    """Return the latest user message content, or an empty string."""
    for message in reversed(messages):
        if message.get("role") == "user" and isinstance(message.get("content"), str):
            return str(message["content"])
    return ""


def normalize_language_code(language: str) -> str:
    """Return the primary subtag of a language code, lowercased."""
    return language.casefold().split("-", 1)[0]


def safe_log_token(value: str, maximum_length: int = 128) -> str:
    """Keep model- or client-supplied identifiers on one safe log line."""
    sanitized = "".join(
        character
        if character.isascii() and (character.isalnum() or character in "._-")
        else "_"
        for character in value
    )
    return sanitized[:maximum_length] or "-"
