import json
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

import pytest

from home_cortex.agents import get_agent
from home_cortex.calendar import (
    CalendarBinding,
    CalendarService,
    CalendarUnavailableError,
    GoogleCalendarProvider,
    ProviderEvent,
    normalize_google_event,
)
from home_cortex.tools import ToolDispatcher, tool_caller_scope
from test_agent import FakeOllamaService, _agent, _chat_response, _tool_call
from test_tools import FakeRetrievalService

PACIFIC = ZoneInfo("America/Los_Angeles")
STEWARD = get_agent("steward")


class FakeCalendarProvider:
    def __init__(
        self,
        events: list[ProviderEvent] | None = None,
        *,
        error: Exception | None = None,
        errors_by_calendar: dict[str, Exception] | None = None,
    ) -> None:
        self.events = events or []
        self.error = error
        self.errors_by_calendar = errors_by_calendar or {}
        self.calls: list[tuple[str, datetime, datetime, int]] = []

    async def list_events(
        self,
        provider_calendar_id: str,
        start: datetime,
        end: datetime,
        *,
        limit: int,
    ) -> list[ProviderEvent]:
        self.calls.append((provider_calendar_id, start, end, limit))
        if provider_calendar_id in self.errors_by_calendar:
            raise self.errors_by_calendar[provider_calendar_id]
        if self.error is not None:
            raise self.error
        return [
            event
            for event in self.events
            if event.start < end and event.end > start
        ][:limit]


def _event(
    event_id: str = "evt-1",
    *,
    title: str = "Dentist",
    start: str = "2026-08-23T09:00:00-07:00",
    end: str = "2026-08-23T10:00:00-07:00",
    busy: bool = True,
    all_day: bool = False,
    timezone_name: str = "America/Los_Angeles",
) -> ProviderEvent:
    return ProviderEvent(
        provider_event_id=event_id,
        title=title,
        start=datetime.fromisoformat(start),
        end=datetime.fromisoformat(end),
        timezone=timezone_name,
        all_day=all_day,
        status="confirmed",
        location="Clinic",
        busy=busy,
    )


def _service(
    provider: FakeCalendarProvider,
    *,
    readers: tuple[str, ...] = (),
) -> CalendarService:
    return CalendarService(
        [
            CalendarBinding(
                id="jian_primary",
                person_id="person:jian_kuang",
                provider_calendar_id="primary",
                readers=readers,
            ),
            CalendarBinding(
                id="pu_primary",
                person_id="person:pu_ba",
                provider_calendar_id="pu@example.com",
            ),
        ],
        provider,
        default_timezone="America/Los_Angeles",
    )


def _dispatcher(
    provider: FakeCalendarProvider | None = None,
    **service_kwargs: Any,
) -> tuple[ToolDispatcher, FakeCalendarProvider]:
    calendar_provider = provider or FakeCalendarProvider(events=[_event()])
    dispatcher = ToolDispatcher(
        FakeRetrievalService(),  # type: ignore[arg-type]
        calendar=_service(calendar_provider, **service_kwargs),
    )
    return dispatcher, calendar_provider


@pytest.mark.asyncio
async def test_list_events_preserves_timezone_and_window() -> None:
    dispatcher, provider = _dispatcher()

    with tool_caller_scope("person:jian_kuang"):
        response = await dispatcher.dispatch(
            "calendar.list_events",
            {"start": "2026-08-23", "end": "2026-08-24"},
        )

    assert response["ok"] is True
    payload = response["result"]
    assert payload["timezone"] == "America/Los_Angeles"
    assert payload["start"] == "2026-08-23T00:00:00-07:00"
    assert payload["end"] == "2026-08-24T00:00:00-07:00"
    assert payload["calendars"] == ["jian_primary"]
    assert payload["events"][0]["start"] == "2026-08-23T09:00:00-07:00"
    assert payload["events"][0]["timezone"] == "America/Los_Angeles"
    assert payload["events"][0]["title"] == "Dentist"
    serialized = json.dumps(payload)
    assert "provider_calendar_id" not in serialized
    assert "pu@example.com" not in serialized
    assert provider.calls[0][0] == "primary"
    assert provider.calls[0][1] == datetime(2026, 8, 23, tzinfo=PACIFIC)


@pytest.mark.asyncio
async def test_check_availability_reports_conflicts_and_free_windows() -> None:
    dispatcher, _ = _dispatcher()

    with tool_caller_scope("person:jian_kuang"):
        busy = await dispatcher.dispatch(
            "calendar.check_availability",
            {
                "start": "2026-08-23T09:00:00-07:00",
                "end": "2026-08-23T09:30:00-07:00",
            },
        )
        free = await dispatcher.dispatch(
            "calendar.check_availability",
            {
                "start": "2026-08-23T14:00:00-07:00",
                "end": "2026-08-23T15:00:00-07:00",
            },
        )

    assert busy["result"]["available"] is False
    assert busy["result"]["conflicts"][0]["title"] == "Dentist"
    assert free["result"]["available"] is True
    assert free["result"]["conflicts"] == []


@pytest.mark.asyncio
async def test_unauthorized_person_or_calendar_fails_closed() -> None:
    dispatcher, provider = _dispatcher()

    with tool_caller_scope("person:jian_kuang"):
        other_person = await dispatcher.dispatch(
            "calendar.list_events",
            {
                "start": "2026-08-23",
                "end": "2026-08-24",
                "person": "person:pu_ba",
            },
        )
        other_calendar = await dispatcher.dispatch(
            "calendar.list_events",
            {
                "start": "2026-08-23",
                "end": "2026-08-24",
                "calendar": "pu_primary",
            },
        )

    with tool_caller_scope(None):
        anonymous = await dispatcher.dispatch(
            "calendar.check_availability",
            {"start": "2026-08-23T14:00:00", "end": "2026-08-23T15:00:00"},
        )

    assert other_person["error"]["code"] == "unauthorized"
    assert other_calendar["error"]["code"] == "unauthorized"
    assert anonymous["error"]["code"] == "unauthorized"
    assert provider.calls == []


@pytest.mark.asyncio
async def test_explicit_reader_grant_allows_another_household_calendar() -> None:
    provider = FakeCalendarProvider(events=[_event()])
    service = CalendarService(
        [
            CalendarBinding(
                id="pu_primary",
                person_id="person:pu_ba",
                provider_calendar_id="pu@example.com",
                readers=("person:jian_kuang",),
            )
        ],
        provider,
        default_timezone="America/Los_Angeles",
    )
    dispatcher = ToolDispatcher(
        FakeRetrievalService(),  # type: ignore[arg-type]
        calendar=service,
    )

    with tool_caller_scope("person:jian_kuang"):
        response = await dispatcher.dispatch(
            "calendar.list_events",
            {
                "start": "2026-08-23",
                "end": "2026-08-24",
                "person": "person:pu_ba",
            },
        )

    assert response["ok"] is True
    assert response["result"]["calendars"] == ["pu_primary"]


@pytest.mark.asyncio
async def test_provider_errors_do_not_crash_and_stay_privacy_safe() -> None:
    dispatcher, _ = _dispatcher(
        FakeCalendarProvider(
            error=CalendarUnavailableError(
                "token ya29.secret should never leak"
            )
        )
    )

    with tool_caller_scope("person:jian_kuang"):
        response = await dispatcher.dispatch(
            "calendar.list_events",
            {"start": "2026-08-23", "end": "2026-08-24"},
        )

    assert response["ok"] is False
    assert response["error"]["code"] == "calendar_unavailable"
    assert response["error"]["message"] == (
        "The calendar service is temporarily unavailable"
    )
    serialized = json.dumps(response)
    assert "ya29.secret" not in serialized
    assert "refresh_token" not in serialized
    assert "client_secret" not in serialized


@pytest.mark.asyncio
async def test_unavailable_calendar_is_skipped_when_another_succeeds() -> None:
    provider = FakeCalendarProvider(
        events=[_event()],
        errors_by_calendar={
            "family@example.com": CalendarUnavailableError("gone"),
        },
    )
    service = CalendarService(
        [
            CalendarBinding(
                id="family",
                person_id="person:jian_kuang",
                provider_calendar_id="family@example.com",
            ),
            CalendarBinding(
                id="jian_primary",
                person_id="person:jian_kuang",
                provider_calendar_id="primary",
            ),
        ],
        provider,
        default_timezone="America/Los_Angeles",
    )
    dispatcher = ToolDispatcher(
        FakeRetrievalService(),  # type: ignore[arg-type]
        calendar=service,
    )

    with tool_caller_scope("person:jian_kuang"):
        response = await dispatcher.dispatch(
            "calendar.list_events",
            {"start": "2026-08-23", "end": "2026-08-24"},
        )

    assert response["ok"] is True
    assert response["result"]["unavailable_calendars"] == ["family"]
    assert response["result"]["events"][0]["calendar_id"] == "jian_primary"


@pytest.mark.asyncio
async def test_unconfigured_calendar_service_fails_closed() -> None:
    dispatcher = ToolDispatcher(FakeRetrievalService())  # type: ignore[arg-type]

    with tool_caller_scope("person:jian_kuang"):
        response = await dispatcher.dispatch(
            "calendar.list_events",
            {"start": "2026-08-23", "end": "2026-08-24"},
        )

    assert response["error"]["code"] == "calendar_unavailable"


def test_google_event_normalization_keeps_offsets_and_all_day_dates() -> None:
    timed = normalize_google_event(
        {
            "id": "timed-1",
            "summary": "School pickup",
            "start": {
                "dateTime": "2026-08-23T15:00:00-07:00",
                "timeZone": "America/Los_Angeles",
            },
            "end": {
                "dateTime": "2026-08-23T15:30:00-07:00",
                "timeZone": "America/Los_Angeles",
            },
            "status": "confirmed",
        },
        "America/Los_Angeles",
    )
    all_day = normalize_google_event(
        {
            "id": "all-day-1",
            "summary": "Holiday",
            "start": {"date": "2026-08-23"},
            "end": {"date": "2026-08-24"},
            "transparency": "transparent",
        },
        "America/Los_Angeles",
    )
    cancelled = normalize_google_event(
        {"id": "cancelled-1", "status": "cancelled", "summary": "Nope"},
        "America/Los_Angeles",
    )

    assert timed is not None
    assert timed.start.isoformat() == "2026-08-23T15:00:00-07:00"
    assert timed.timezone == "America/Los_Angeles"
    assert all_day is not None
    assert all_day.all_day is True
    assert all_day.busy is False
    assert all_day.start.tzinfo == PACIFIC
    assert cancelled is None


@pytest.mark.asyncio
async def test_google_provider_does_not_expose_credentials_on_http_failure() -> None:
    def http_json(method: str, url: str, headers: dict[str, str], data: bytes | None):
        if method == "POST":
            return 200, {"access_token": "ya29.secret-token", "expires_in": 3600}
        return 500, {"error": "refresh_token leaked in provider body"}

    provider = GoogleCalendarProvider(
        client_id="client-id",
        client_secret="client-secret",
        refresh_token="refresh-token",
        default_timezone="America/Los_Angeles",
        http_json=http_json,
    )

    with pytest.raises(CalendarUnavailableError, match="temporarily unavailable") as error:
        await provider.list_events(
            "primary",
            datetime(2026, 8, 23, tzinfo=PACIFIC),
            datetime(2026, 8, 24, tzinfo=PACIFIC),
            limit=10,
        )

    message = str(error.value)
    assert "refresh-token" not in message
    assert "client-secret" not in message
    assert "ya29.secret-token" not in message
    assert "refresh_token leaked" not in message
    assert "credentials=redacted" in repr(provider)


@pytest.mark.asyncio
async def test_butler_answers_tomorrow_from_calendar() -> None:
    dispatcher, _ = _dispatcher()
    ollama = FakeOllamaService(
        [
            _chat_response(
                tool_calls=[
                    _tool_call(
                        "calendar.list_events",
                        {"start": "2026-08-23", "end": "2026-08-24"},
                    )
                ]
            ),
            _chat_response("You have a dentist appointment at 9:00 tomorrow."),
        ]
    )

    result = await _agent(ollama, dispatcher).answer(
        "What do I have tomorrow?",
        user_entity={"id": "person:jian_kuang", "name": ["Jian Kuang"]},
    )
    tool_result = json.loads(ollama.calls[1][-1]["content"])

    assert result.answer == "You have a dentist appointment at 9:00 tomorrow."
    assert result.stop_reason == "answer"
    assert tool_result["ok"] is True
    assert tool_result["result"]["events"][0]["title"] == "Dentist"
    assert tool_result["result"]["events"][0]["start"] == "2026-08-23T09:00:00-07:00"
    assert ollama.tool_names[0] == STEWARD.allowed_tools


@pytest.mark.asyncio
async def test_butler_checks_whether_a_window_is_free() -> None:
    dispatcher, _ = _dispatcher()
    ollama = FakeOllamaService(
        [
            _chat_response(
                tool_calls=[
                    _tool_call(
                        "calendar.check_availability",
                        {
                            "start": "2026-08-23T14:00:00-07:00",
                            "end": "2026-08-23T15:00:00-07:00",
                        },
                    )
                ]
            ),
            _chat_response("That window is free."),
        ]
    )

    result = await _agent(ollama, dispatcher).answer(
        "Am I free tomorrow from 2 to 3pm?",
        user_entity={"id": "person:jian_kuang", "name": ["Jian Kuang"]},
    )
    tool_result = json.loads(ollama.calls[1][-1]["content"])

    assert result.answer == "That window is free."
    assert tool_result["result"]["available"] is True


@pytest.mark.asyncio
async def test_calendar_provider_error_does_not_crash_agent_loop() -> None:
    dispatcher, _ = _dispatcher(
        FakeCalendarProvider(error=RuntimeError("access_token=ya29.secret"))
    )
    ollama = FakeOllamaService(
        [
            _chat_response(
                tool_calls=[
                    _tool_call(
                        "calendar.list_events",
                        {"start": "2026-08-23", "end": "2026-08-24"},
                    )
                ]
            ),
            _chat_response("I could not reach the household calendar."),
        ]
    )

    result = await _agent(ollama, dispatcher).answer(
        "What do I have tomorrow?",
        user_entity={"id": "person:jian_kuang"},
    )
    tool_result = json.loads(ollama.calls[1][-1]["content"])

    assert result.answer == "I could not reach the household calendar."
    assert tool_result["ok"] is False
    assert tool_result["error"]["code"] == "tool_execution_failed"
    assert "ya29.secret" not in json.dumps(tool_result)


@pytest.mark.asyncio
async def test_calculate_is_available_to_the_butler() -> None:
    dispatcher, _ = _dispatcher()
    ollama = FakeOllamaService(
        [
            _chat_response(
                tool_calls=[_tool_call("calculate", {"expression": "2 + 3 * 4"})]
            ),
            _chat_response("The exact result is 14."),
        ]
    )

    result = await _agent(ollama, dispatcher).answer("What is 2 + 3 * 4?")
    tool_result = json.loads(ollama.calls[1][-1]["content"])

    assert result.answer == "The exact result is 14."
    assert tool_result["result"]["result"] == 14
