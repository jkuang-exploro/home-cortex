"""GUI session routes."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request
from starlette.responses import JSONResponse

from ...conversation.session import (
    COOKIE_NAME as GUI_COOKIE_NAME,
    SESSION_SECONDS as GUI_SESSION_SECONDS,
    parse_session,
    session_token as gui_session_token,
)
from ...common.identity import OPENWEBUI_USER_EMAIL_HEADER, OPENWEBUI_USER_ID_HEADER
from ..dependencies import (
    authenticate_bearer,
    authenticate_request,
    request_settings,
    session_identity,
)
from ..schemas import SessionRequest


router = APIRouter()


@router.post("/session")
async def create_session(body: SessionRequest, request: Request) -> JSONResponse:
    authenticate_bearer(request)
    settings = request_settings(request)
    kind, value = session_identity(body, settings)
    payload: dict[str, Any] = {"object": "session"}
    if kind == "email":
        payload["email"] = value
    elif kind == "id":
        payload["user_id"] = value
    response = JSONResponse(payload)
    key = settings.cortex_api_key
    if key:
        response.set_cookie(
            GUI_COOKIE_NAME,
            gui_session_token(key, kind, value),
            max_age=GUI_SESSION_SECONDS,
            httponly=True,
            secure=request.url.scheme == "https",
            samesite="strict",
            path="/",
        )
    return response


@router.get("/session")
async def read_session(request: Request) -> dict[str, Any]:
    authenticate_request(request)
    settings = request_settings(request)
    key = settings.cortex_api_key
    parsed = parse_session(request.cookies.get(GUI_COOKIE_NAME, ""), key) if key else None
    payload: dict[str, Any] = {"object": "session"}
    if parsed is None:
        if key is None:
            payload["anonymous"] = True
            return payload
        user_id = request.headers.get(OPENWEBUI_USER_ID_HEADER)
        email = request.headers.get(OPENWEBUI_USER_EMAIL_HEADER)
        if user_id:
            payload["user_id"] = user_id
        if email:
            payload["email"] = email
        if not user_id and not email:
            payload["anonymous"] = True
        return payload
    kind, value = parsed
    if kind == "email":
        payload["email"] = value
    elif kind == "id":
        payload["user_id"] = value
    else:
        payload["anonymous"] = True
    return payload


@router.delete("/session")
async def delete_session() -> JSONResponse:
    response = JSONResponse({"status": "ok"})
    response.delete_cookie(GUI_COOKIE_NAME, path="/")
    return response
