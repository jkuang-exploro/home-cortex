import re
from typing import Any


TABLE_NAME_PATTERN = r"[A-Za-z_][A-Za-z0-9_]*"
RECORD_KEY_PATTERN = r"[A-Za-z0-9_-]+(?::[A-Za-z0-9_-]+)*"
RECORD_ID_PATTERN = rf"^{TABLE_NAME_PATTERN}:{RECORD_KEY_PATTERN}$"

TABLE_NAME_RE = re.compile(rf"^{TABLE_NAME_PATTERN}$")
RECORD_ID_RE = re.compile(
    rf"^(?P<table>{TABLE_NAME_PATTERN}):(?P<id>{RECORD_KEY_PATTERN})$"
)


def split_record_id(value: str) -> tuple[str, str]:
    """Split a canonical graph ID, including IDs with colon-delimited segments."""
    if not isinstance(value, str):
        raise ValueError("Entity ID must use the table:record_id format")
    match = RECORD_ID_RE.fullmatch(value)
    if match is None:
        raise ValueError("Entity ID must use the table:record_id format")
    return match.group("table"), match.group("id")


def canonical_record_id(value: Any) -> str:
    """Return the public table:key form for a SurrealDB RecordID-like value."""
    table = getattr(value, "table_name", None)
    record_id = getattr(value, "id", None)
    if table is None or record_id is None:
        return str(value)
    return f"{table}:{record_id}"
