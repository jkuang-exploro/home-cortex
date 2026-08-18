"""Map trusted Open WebUI metadata to a home-graph Person ID.

V1 authenticates the household client with a shared API key. That key does
not bind a person. Person identity comes only from X-OpenWebUI-User-Id or
X-OpenWebUI-User-Email through CORTEX_IDENTITY_MAP. A client-supplied
person record ID is never treated as identity.
"""

from collections.abc import Mapping

OPENWEBUI_USER_ID_HEADER = "X-OpenWebUI-User-Id"
OPENWEBUI_USER_EMAIL_HEADER = "X-OpenWebUI-User-Email"


def resolve_user_entity_id(
    headers: Mapping[str, str],
    identity_map: Mapping[str, str],
) -> str | None:
    """Resolve trusted Open WebUI metadata to a home-graph person ID."""
    user_id = headers.get(OPENWEBUI_USER_ID_HEADER)
    if user_id:
        entity_id = identity_map.get(f"id:{user_id.strip()}")
        if entity_id:
            return entity_id

    email = headers.get(OPENWEBUI_USER_EMAIL_HEADER)
    if email:
        return identity_map.get(f"email:{email.strip().casefold()}")
    return None
