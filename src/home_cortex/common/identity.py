"""Map trusted client metadata to a home-graph Person ID.

V1 authenticates the household client with a shared API key. That key does
not bind a person. Person identity comes from X-OpenWebUI-User-Id /
X-OpenWebUI-User-Email or from a GUI session that stores the same map keys,
through CORTEX_IDENTITY_MAP. A client-supplied person record ID is never
treated as identity.
"""

from collections.abc import Mapping

OPENWEBUI_USER_ID_HEADER = "X-OpenWebUI-User-Id"
OPENWEBUI_USER_EMAIL_HEADER = "X-OpenWebUI-User-Email"


def resolve_user_entity_id(
    headers: Mapping[str, str],
    identity_map: Mapping[str, str],
    *,
    user_id: str | None = None,
    email: str | None = None,
) -> str | None:
    """Resolve trusted user id/email metadata to a home-graph person ID."""
    header_id = headers.get(OPENWEBUI_USER_ID_HEADER) or user_id
    if header_id:
        entity_id = identity_map.get(f"id:{header_id.strip()}")
        if entity_id:
            return entity_id

    header_email = headers.get(OPENWEBUI_USER_EMAIL_HEADER) or email
    if header_email:
        return identity_map.get(f"email:{header_email.strip().casefold()}")
    return None
