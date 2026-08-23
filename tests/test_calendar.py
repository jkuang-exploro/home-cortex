import asyncio
import json
from datetime import datetime
from typing import Any
from urllib.parse import parse_qs, urlparse
from zoneinfo import ZoneInfo

import pytest

from home_cortex.agents import get_agent
from home_cortex.calendar import (
    CalendarBinding,
    CalendarService,
    CalendarUnavailableError,
    GoogleCalendarProvider,
    ProviderEvent,
    ProviderEventBatch,
    normalize_google_event,
)
from home_cortex.tools import ToolDispatcher
from test_agent_service import (
    FakeDispatcher,
    FakeOllamaService,
    _agent,
    _chat_response,
    _tool_call,
)
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
        events_by_calendar: dict[str, list[ProviderEvent]] | None = None,
        delay: float = 0,
    ) -> None:
        self.events = events or []
        self.error = error
        self.errors_by_calendar = errors_by_calendar or {}
        self.events_by_calendar = events_by_calendar or {}
        self.delay = delay
        self.calls: list[tuple[str, datetime, datetime, int]] = []

    async def list_events(
        self,
        provider_calendar_id: str,
        start: datetime,
        end: datetime,
        *,
        limit: int,
    ) -> ProviderEventBatch:
        self.calls.append((provider_calendar_id, start, end, limit))
        if self.delay:
            await asyncio.sleep(self.delay)
        if provider_calendar_id in self.errors_by_calendar:
            raise self.errors_by_calendar[provider_calendar_id]
        if self.error is not None:
            raise self.error
        pool = self.events_by_calendar.get(provider_calendar_id, self.events)
        matching = [
            event
            for event in pool
            if event.start < end and event.end > start
        ]
        return ProviderEventBatch(
            tuple(matching[:limit]),
            complete=len(matching) <= limit,
        )


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


async def _call(
    dispatcher: ToolDispatcher,
    tool_name: str,
    arguments: dict[str, Any],
    *,
    caller: str | None = "person:jian_kuang",
) -> dict[str, Any]:
    return await dispatcher.dispatch(
        tool_name,
        arguments,
        caller_entity_id=caller,
    )


@pytest.mark.asyncio
async def test_list_events_preserves_timezone_and_window() -> None:
    dispatcher, provider = _dispatcher()

    response = await _call(
        dispatcher,
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

    busy = await _call(
        dispatcher,
        "calendar.check_availability",
        {
            "start": "2026-08-23T09:00:00-07:00",
            "end": "2026-08-23T09:30:00-07:00",
        },
    )
    free = await _call(
        dispatcher,
        "calendar.check_availability",
        {
            "start": "2026-08-23T14:00:00-07:00",
            "end": "2026-08-23T15:00:00-07:00",
        },
    )

    assert busy["result"]["available"] is False
    assert busy["result"]["checked"] is True
    assert busy["result"]["conflicts"][0]["title"] == "Dentist"
    assert free["result"]["available"] is True
    assert free["result"]["checked"] is True
    assert free["result"]["conflicts"] == []


@pytest.mark.asyncio
async def test_unauthorized_person_or_calendar_fails_closed() -> None:
    dispatcher, provider = _dispatcher()

    other_person = await _call(
        dispatcher,
        "calendar.list_events",
        {
            "start": "2026-08-23",
            "end": "2026-08-24",
            "person": "person:pu_ba",
        },
    )
    other_calendar = await _call(
        dispatcher,
        "calendar.list_events",
        {
            "start": "2026-08-23",
            "end": "2026-08-24",
            "calendar": "pu_primary",
        },
    )
    anonymous = await _call(
        dispatcher,
        "calendar.check_availability",
        {"start": "2026-08-23T14:00:00", "end": "2026-08-23T15:00:00"},
        caller=None,
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

    response = await _call(
        dispatcher,
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

    response = await _call(
        dispatcher,
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

    response = await _call(
        dispatcher,
        "calendar.list_events",
        {"start": "2026-08-23", "end": "2026-08-24"},
    )

    assert response["ok"] is True
    assert response["result"]["unavailable_calendars"] == ["family"]
    assert response["result"]["calendars"] == ["jian_primary"]
    assert response["result"]["events"][0]["calendar_id"] == "jian_primary"


@pytest.mark.asyncio
async def test_timed_out_calendar_does_not_discard_a_successful_calendar() -> None:
    class SlowFamilyProvider(FakeCalendarProvider):
        async def list_events(
            self,
            provider_calendar_id: str,
            start: datetime,
            end: datetime,
            *,
            limit: int,
        ) -> ProviderEventBatch:
            if provider_calendar_id == "family@example.com":
                await asyncio.sleep(0.05)
            return await super().list_events(
                provider_calendar_id,
                start,
                end,
                limit=limit,
            )

    provider = SlowFamilyProvider(events=[_event()])
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
        binding_timeout_seconds=0.01,
    )
    dispatcher = ToolDispatcher(
        FakeRetrievalService(),  # type: ignore[arg-type]
        calendar=service,
    )

    response = await _call(
        dispatcher,
        "calendar.list_events",
        {"start": "2026-08-23", "end": "2026-08-24"},
    )

    assert response["ok"] is True
    assert response["result"]["complete"] is False
    assert response["result"]["calendars"] == ["jian_primary"]
    assert response["result"]["unavailable_calendars"] == ["family"]
    assert response["result"]["events"][0]["calendar_id"] == "jian_primary"


@pytest.mark.asyncio
async def test_unconfigured_calendar_service_fails_closed() -> None:
    dispatcher = ToolDispatcher(FakeRetrievalService())  # type: ignore[arg-type]

    response = await _call(
        dispatcher,
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
async def test_google_provider_follows_pages_and_reports_a_remaining_page() -> None:
    get_queries: list[dict[str, list[str]]] = []

    def event(event_id: str, hour: int) -> dict[str, Any]:
        return {
            "id": event_id,
            "summary": event_id,
            "start": {"dateTime": f"2026-08-23T{hour:02d}:00:00-07:00"},
            "end": {"dateTime": f"2026-08-23T{hour + 1:02d}:00:00-07:00"},
        }

    def http_json(method: str, url: str, headers: dict[str, str], data: bytes | None):
        if method == "POST":
            return 200, {"access_token": "token", "expires_in": 3600}
        query = parse_qs(urlparse(url).query)
        get_queries.append(query)
        if query.get("pageToken") == ["page-2"]:
            return 200, {"items": [event("second", 10)]}
        return 200, {
            "items": [event("first", 9)],
            "nextPageToken": "page-2",
        }

    provider = GoogleCalendarProvider(
        client_id="client-id",
        client_secret="client-secret",
        refresh_token="refresh-token",
        default_timezone="America/Los_Angeles",
        http_json=http_json,
    )
    start = datetime(2026, 8, 23, tzinfo=PACIFIC)
    end = datetime(2026, 8, 24, tzinfo=PACIFIC)

    complete = await provider.list_events("primary", start, end, limit=2)
    incomplete = await provider.list_events("primary", start, end, limit=1)

    assert [item.provider_event_id for item in complete.events] == [
        "first",
        "second",
    ]
    assert complete.complete is True
    assert incomplete.complete is False
    assert [item.provider_event_id for item in incomplete.events] == ["first"]
    assert get_queries[1]["pageToken"] == ["page-2"]


@pytest.mark.asyncio
async def test_google_provider_keeps_completed_pages_when_a_later_page_fails() -> None:
    get_count = 0

    def http_json(
        method: str,
        url: str,
        headers: dict[str, str],
        data: bytes | None,
    ) -> tuple[int, Any]:
        nonlocal get_count
        if method == "POST":
            return 200, {"access_token": "token", "expires_in": 3600}
        get_count += 1
        if get_count == 1:
            return 200, {
                "items": [
                    {
                        "id": "first",
                        "summary": "First",
                        "start": {"dateTime": "2026-08-23T09:00:00-07:00"},
                        "end": {"dateTime": "2026-08-23T10:00:00-07:00"},
                    }
                ],
                "nextPageToken": "page-2",
            }
        return 500, {"error": "temporarily unavailable"}

    provider = GoogleCalendarProvider(
        client_id="client-id",
        client_secret="client-secret",
        refresh_token="refresh-token",
        default_timezone="America/Los_Angeles",
        http_json=http_json,
    )

    result = await provider.list_events(
        "primary",
        datetime(2026, 8, 23, tzinfo=PACIFIC),
        datetime(2026, 8, 24, tzinfo=PACIFIC),
        limit=10,
    )

    assert [event.provider_event_id for event in result.events] == ["first"]
    assert result.complete is False


@pytest.mark.asyncio
async def test_google_provider_skips_401_retry_that_cannot_fit_budget() -> None:
    methods: list[str] = []

    def http_json(
        method: str,
        url: str,
        headers: dict[str, str],
        data: bytes | None,
    ) -> tuple[int, Any]:
        methods.append(method)
        if method == "POST":
            return 200, {"access_token": "token", "expires_in": 3600}
        return 401, {"error": "expired"}

    provider = GoogleCalendarProvider(
        client_id="client-id",
        client_secret="client-secret",
        refresh_token="refresh-token",
        default_timezone="America/Los_Angeles",
        http_json=http_json,
        fetch_budget_seconds=1.6,
    )

    result = await provider.list_events(
        "primary",
        datetime(2026, 8, 23, tzinfo=PACIFIC),
        datetime(2026, 8, 24, tzinfo=PACIFIC),
        limit=10,
    )

    assert result.events == ()
    assert result.complete is False
    assert methods == ["POST", "GET"]


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


@pytest.mark.asyncio
async def test_same_day_date_only_range_covers_that_household_day() -> None:
    dispatcher, provider = _dispatcher()

    response = await _call(
        dispatcher,
        "calendar.list_events",
        {"start": "2026-08-23", "end": "2026-08-23"},
    )

    assert response["ok"] is True
    assert response["result"]["start"] == "2026-08-23T00:00:00-07:00"
    assert response["result"]["end"] == "2026-08-24T00:00:00-07:00"
    assert response["result"]["events"][0]["title"] == "Dentist"
    assert provider.calls[0][2] == datetime(2026, 8, 24, tzinfo=PACIFIC)


@pytest.mark.asyncio
async def test_events_sort_chronologically_across_offsets() -> None:
    dispatcher, _ = _dispatcher(
        FakeCalendarProvider(
            events=[
                _event(
                    "later-pacific",
                    title="Later Pacific",
                    start="2026-08-23T10:00:00-07:00",
                    end="2026-08-23T11:00:00-07:00",
                ),
                _event(
                    "earlier-utc",
                    title="Earlier UTC",
                    start="2026-08-23T16:00:00+00:00",
                    end="2026-08-23T17:00:00+00:00",
                    timezone_name="UTC",
                ),
            ]
        )
    )

    response = await _call(
        dispatcher,
        "calendar.list_events",
        {"start": "2026-08-23", "end": "2026-08-24"},
    )

    titles = [event["title"] for event in response["result"]["events"]]
    assert titles == ["Earlier UTC", "Later Pacific"]


@pytest.mark.asyncio
async def test_later_calendar_events_are_not_starved_by_earlier_quota() -> None:
    family_events = [
        _event(
            f"family-{index}",
            title=f"Family {index}",
            start=f"2026-08-23T{12 + index:02d}:00:00-07:00",
            end=f"2026-08-23T{12 + index:02d}:30:00-07:00",
        )
        for index in range(10)
    ]
    provider = FakeCalendarProvider(
        events_by_calendar={
            "family@example.com": family_events,
            "primary": [
                _event(
                    "jian-early",
                    title="Early personal",
                    start="2026-08-23T08:00:00-07:00",
                    end="2026-08-23T08:30:00-07:00",
                )
            ],
        }
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

    response = await _call(
        dispatcher,
        "calendar.list_events",
        {"start": "2026-08-23", "end": "2026-08-24", "limit": 5},
    )

    titles = [event["title"] for event in response["result"]["events"]]
    assert titles[0] == "Early personal"
    assert len(titles) == 5
    assert "Family 0" in titles


@pytest.mark.asyncio
async def test_mixed_provider_failure_does_not_report_available() -> None:
    provider = FakeCalendarProvider(
        events=[],
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

    response = await _call(
        dispatcher,
        "calendar.check_availability",
        {
            "start": "2026-08-23T14:00:00-07:00",
            "end": "2026-08-23T15:00:00-07:00",
        },
    )

    assert response["ok"] is True
    assert response["result"]["checked"] is False
    assert response["result"]["available"] is False
    assert response["result"]["unavailable_calendars"] == ["family"]


@pytest.mark.asyncio
async def test_truncated_provider_page_does_not_report_available() -> None:
    transparent = [
        _event(
            f"transparent-{index}",
            title="Transparent",
            busy=False,
        )
        for index in range(100)
    ]
    provider = FakeCalendarProvider(
        events=[*transparent, _event("busy-after-first-page", title="Conflict")]
    )
    service = CalendarService(
        [
            CalendarBinding(
                id="jian_primary",
                person_id="person:jian_kuang",
                provider_calendar_id="primary",
            )
        ],
        provider,
        default_timezone="America/Los_Angeles",
    )
    dispatcher = ToolDispatcher(
        FakeRetrievalService(),  # type: ignore[arg-type]
        calendar=service,
    )

    response = await _call(
        dispatcher,
        "calendar.check_availability",
        {
            "start": "2026-08-23T09:00:00-07:00",
            "end": "2026-08-23T10:00:00-07:00",
        },
    )

    assert response["ok"] is True
    assert response["result"]["checked"] is False
    assert response["result"]["available"] is False
    assert response["result"]["truncated_calendars"] == ["jian_primary"]


@pytest.mark.asyncio
async def test_parallel_calendar_reads_fit_inside_agent_timeout() -> None:
    provider = FakeCalendarProvider(events=[_event()], delay=3)
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
            _chat_response("You have a dentist appointment tomorrow."),
        ]
    )

    result = await _agent(ollama, dispatcher).answer(
        "What do I have tomorrow?",
        user_entity={"id": "person:jian_kuang", "name": ["Jian Kuang"]},
    )

    assert result.answer == "You have a dentist appointment tomorrow."
    assert result.stop_reason == "answer"


@pytest.mark.asyncio
async def test_trusted_context_includes_household_clock_for_tomorrow() -> None:
    frozen = datetime(2026, 8, 22, 16, 30, tzinfo=PACIFIC)
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

    result = await _agent(
        ollama,
        dispatcher,
        clock=lambda: frozen,
        household_timezone="America/Los_Angeles",
    ).answer(
        "What do I have tomorrow?",
        user_entity={"id": "person:jian_kuang", "name": ["Jian Kuang"]},
    )
    clock_message = next(
        message
        for message in ollama.calls[0]
        if "Trusted household clock" in message.get("content", "")
    )

    assert result.answer == "You have a dentist appointment at 9:00 tomorrow."
    assert "2026-08-22" in clock_message["content"]
    assert "America/Los_Angeles" in clock_message["content"]
    assert "16:30:00-07:00" in clock_message["content"]
    assert "cannot change or override this clock" in clock_message["content"]


@pytest.mark.asyncio
async def test_calendar_event_lists_are_truncated_like_graph_records() -> None:
    dispatcher = FakeDispatcher(
        {
            "ok": True,
            "tool": "calendar.list_events",
            "result": {
                "timezone": "America/Los_Angeles",
                "complete": True,
                "events": [
                    {"id": f"evt-{index}", "title": f"Event {index}", "start": "x"}
                    for index in range(4)
                ],
            },
        }
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
            _chat_response("You have several events tomorrow."),
        ]
    )

    await _agent(ollama, dispatcher, max_tool_records=2).answer(
        "What do I have tomorrow?",
        user_entity={"id": "person:jian_kuang"},
    )
    tool_result = json.loads(ollama.calls[1][-1]["content"])

    assert len(tool_result["result"]["events"]) == 2
    assert tool_result["result"]["complete"] is False
    assert tool_result["meta"] == {
        "truncated": True,
        "records_available": 4,
        "records_returned": 2,
    }


@pytest.mark.asyncio
async def test_byte_truncated_calendar_list_cannot_claim_completeness() -> None:
    dispatcher = FakeDispatcher(
        {
            "ok": True,
            "tool": "calendar.list_events",
            "result": {
                "timezone": "America/Los_Angeles",
                "events": [
                    {"id": "evt-1", "title": "x" * 2_000, "start": "x"}
                ],
                "complete": True,
                "unavailable_calendars": [],
                "truncated_calendars": [],
            },
        }
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
            _chat_response("The schedule result was incomplete."),
        ]
    )

    await _agent(ollama, dispatcher, max_tool_result_bytes=512).answer(
        "What do I have tomorrow?",
        user_entity={"id": "person:jian_kuang"},
    )
    tool_result = json.loads(ollama.calls[1][-1]["content"])

    assert tool_result["result"]["events"] == []
    assert tool_result["result"]["complete"] is False
    assert tool_result["meta"]["truncated"] is True


@pytest.mark.asyncio
async def test_truncated_conflicts_fail_closed() -> None:
    dispatcher = FakeDispatcher(
        {
            "ok": True,
            "tool": "calendar.check_availability",
            "result": {
                "timezone": "America/Los_Angeles",
                "checked": True,
                "available": False,
                "conflicts": [
                    {"id": f"evt-{index}", "title": f"Conflict {index}"}
                    for index in range(4)
                ],
                "unavailable_calendars": [],
                "truncated_calendars": [],
            },
        }
    )
    ollama = FakeOllamaService(
        [
            _chat_response(
                tool_calls=[
                    _tool_call(
                        "calendar.check_availability",
                        {
                            "start": "2026-08-23T09:00:00-07:00",
                            "end": "2026-08-23T10:00:00-07:00",
                        },
                    )
                ]
            ),
            _chat_response("The availability result was incomplete."),
        ]
    )

    await _agent(ollama, dispatcher, max_tool_records=2).answer(
        "Am I free tomorrow at 9?",
        user_entity={"id": "person:jian_kuang"},
    )
    tool_result = json.loads(ollama.calls[1][-1]["content"])

    assert len(tool_result["result"]["conflicts"]) == 2
    assert tool_result["result"]["checked"] is False
    assert tool_result["result"]["available"] is False
    assert tool_result["meta"]["truncated"] is True


@pytest.mark.asyncio
async def test_relationship_lookup_can_continue_into_calendar_tool() -> None:
    class ChainedDispatcher:
        def __init__(self) -> None:
            self.calls: list[tuple[str, Any, str | None]] = []

        async def dispatch(
            self,
            tool_name: str,
            arguments: Any,
            *,
            caller_entity_id: str | None = None,
        ) -> dict[str, Any]:
            self.calls.append((tool_name, arguments, caller_entity_id))
            if tool_name == "get_relationships":
                return {
                    "ok": True,
                    "tool": tool_name,
                    "result": [
                        {
                            "id": "parent_of:jian_evelyn",
                            "relation": "parent_of",
                            "related_entity": {
                                "id": "person:evelyn_kuang",
                                "name": ["Evelyn Kuang"],
                                "gender": "female",
                            },
                        }
                    ],
                }
            if tool_name == "calendar.list_events":
                return {
                    "ok": True,
                    "tool": tool_name,
                    "result": {
                        "start": "2026-08-23T00:00:00-07:00",
                        "end": "2026-08-24T00:00:00-07:00",
                        "timezone": "America/Los_Angeles",
                        "calendars": ["evelyn_primary"],
                        "events": [
                            {
                                "id": "evelyn_primary:school",
                                "calendar_id": "evelyn_primary",
                                "person_id": "person:evelyn_kuang",
                                "title": "School",
                                "start": "2026-08-23T08:00:00-07:00",
                                "end": "2026-08-23T15:00:00-07:00",
                                "timezone": "America/Los_Angeles",
                                "all_day": False,
                                "status": "confirmed",
                                "location": None,
                                "busy": True,
                            }
                        ],
                        "complete": True,
                        "unavailable_calendars": [],
                        "truncated_calendars": [],
                    },
                }
            raise AssertionError(f"Unexpected tool {tool_name}")

    dispatcher = ChainedDispatcher()
    ollama = FakeOllamaService(
        [
            _chat_response(
                tool_calls=[
                    _tool_call(
                        "get_relationships",
                        {
                            "entity_id": "person:jian_kuang",
                            "relation": "parent_of",
                            "direction": "out",
                        },
                    )
                ]
            ),
            _chat_response(
                tool_calls=[
                    _tool_call(
                        "calendar.list_events",
                        {
                            "start": "2026-08-23",
                            "end": "2026-08-24",
                            "person": "person:evelyn_kuang",
                        },
                    )
                ]
            ),
            _chat_response("Evelyn has school tomorrow."),
        ]
    )

    result = await _agent(ollama, dispatcher).answer(
        "What is my daughter's schedule tomorrow?",
        user_entity_id="person:jian_kuang",
    )

    assert result.answer == "Evelyn has school tomorrow."
    assert ollama.tool_names[1] == (
        "calculate",
        "calendar.list_events",
        "calendar.check_availability",
    )
    assert [name for name, _, _ in dispatcher.calls] == [
        "get_relationships",
        "calendar.list_events",
    ]
    assert all(
        caller == "person:jian_kuang" for _, _, caller in dispatcher.calls
    )
