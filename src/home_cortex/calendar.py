"""Read-only household calendar access with Cortex-facing normalization."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from time import monotonic
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

PERSON_ID_PATTERN = r"^person:[A-Za-z0-9_-]+$"
CALENDAR_ID_PATTERN = r"^[A-Za-z][A-Za-z0-9_-]*$"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_EVENTS_URL = "https://www.googleapis.com/calendar/v3/calendars/{calendar_id}/events"
HTTP_TIMEOUT_SECONDS = 1.5
GOOGLE_FETCH_BUDGET_SECONDS = 3.5
CALENDAR_BINDING_TIMEOUT_SECONDS = 4.0
TOKEN_EXPIRY_SKEW_SECONDS = 60.0
DEFAULT_EVENT_LIMIT = 25
MAX_EVENT_LIMIT = 100


class CalendarAuthorizationError(PermissionError):
    """Raised when the caller may not access the requested calendars."""


class CalendarUnavailableError(RuntimeError):
    """Raised when the provider cannot complete a read."""


class _CalendarDeadlineExceeded(TimeoutError):
    """Raised internally when another provider request cannot fit the budget."""


@dataclass(frozen=True)
class CalendarBinding:
    id: str
    person_id: str
    provider_calendar_id: str
    readers: tuple[str, ...] = ()

    def readable_by(self, caller_entity_id: str) -> bool:
        return caller_entity_id == self.person_id or caller_entity_id in self.readers


@dataclass(frozen=True)
class ProviderEvent:
    provider_event_id: str
    title: str
    start: datetime
    end: datetime
    timezone: str
    all_day: bool
    status: str
    location: str | None
    busy: bool


@dataclass(frozen=True)
class ProviderEventBatch:
    """A bounded event page plus whether it covers the full query window."""

    events: tuple[ProviderEvent, ...]
    complete: bool


class CalendarProvider(Protocol):
    async def list_events(
        self,
        provider_calendar_id: str,
        start: datetime,
        end: datetime,
        *,
        limit: int,
    ) -> ProviderEventBatch:
        """Return ordered events overlapping [start, end) and completeness."""


class UnconfiguredCalendarProvider:
    async def list_events(
        self,
        provider_calendar_id: str,
        start: datetime,
        end: datetime,
        *,
        limit: int,
    ) -> ProviderEventBatch:
        raise CalendarUnavailableError("The calendar service is not configured")


class CalendarService:
    """Authorize callers and normalize provider events into Cortex schema."""

    def __init__(
        self,
        bindings: Sequence[CalendarBinding],
        provider: CalendarProvider,
        *,
        default_timezone: str,
        binding_timeout_seconds: float = CALENDAR_BINDING_TIMEOUT_SECONDS,
    ) -> None:
        if binding_timeout_seconds <= 0:
            raise ValueError("binding_timeout_seconds must be greater than zero")
        self._bindings = tuple(bindings)
        self._provider = provider
        self._default_timezone = ZoneInfo(default_timezone)
        self._default_timezone_name = default_timezone
        self._binding_timeout_seconds = binding_timeout_seconds
        ids = [binding.id for binding in self._bindings]
        if len(ids) != len(set(ids)):
            raise ValueError("Calendar binding IDs must be unique")

    async def list_events(
        self,
        *,
        start: str,
        end: str,
        calendar_id: str | None,
        person_id: str | None,
        limit: int | None,
        caller_entity_id: str | None,
    ) -> dict[str, Any]:
        window_start, window_end = self._parse_window(start, end)
        selected = self._authorized_calendars(
            caller_entity_id,
            calendar_id=calendar_id,
            person_id=person_id,
        )
        event_limit = DEFAULT_EVENT_LIMIT if limit is None else limit
        loaded = await asyncio.gather(
            *(
                self._load_binding(
                    binding,
                    window_start,
                    window_end,
                    event_limit,
                )
                for binding in selected
            )
        )
        unavailable: list[str] = []
        truncated: list[str] = []
        fetched: list[str] = []
        keyed: list[tuple[datetime, str, dict[str, Any]]] = []
        for binding, provider_events, failed_id, truncated_id in loaded:
            if failed_id is not None:
                unavailable.append(failed_id)
                continue
            fetched.append(binding.id)
            if truncated_id is not None:
                truncated.append(truncated_id)
            for event in provider_events:
                record = _cortex_event(binding, event)
                keyed.append(
                    (_aware(event.start, self._default_timezone), record["id"], record)
                )
        if not fetched and unavailable:
            raise CalendarUnavailableError(
                "The calendar service is temporarily unavailable"
            )
        keyed.sort()
        complete = not unavailable and not truncated and len(keyed) <= event_limit
        return {
            "start": _isoformat(window_start),
            "end": _isoformat(window_end),
            "timezone": self._default_timezone_name,
            "calendars": fetched,
            "events": [record for _, _, record in keyed[:event_limit]],
            "complete": complete,
            "unavailable_calendars": unavailable,
            "truncated_calendars": truncated,
        }

    async def _load_binding(
        self,
        binding: CalendarBinding,
        window_start: datetime,
        window_end: datetime,
        event_limit: int,
    ) -> tuple[
        CalendarBinding,
        tuple[ProviderEvent, ...],
        str | None,
        str | None,
    ]:
        try:
            async with asyncio.timeout(self._binding_timeout_seconds):
                batch = await self._provider.list_events(
                    binding.provider_calendar_id,
                    window_start,
                    window_end,
                    limit=event_limit,
                )
        except (CalendarUnavailableError, TimeoutError):
            return binding, (), binding.id, None
        return (
            binding,
            batch.events,
            None,
            None if batch.complete else binding.id,
        )

    async def check_availability(
        self,
        *,
        start: str,
        end: str,
        calendar_id: str | None,
        person_id: str | None,
        caller_entity_id: str | None,
    ) -> dict[str, Any]:
        listing = await self.list_events(
            start=start,
            end=end,
            calendar_id=calendar_id,
            person_id=person_id,
            limit=MAX_EVENT_LIMIT,
            caller_entity_id=caller_entity_id,
        )
        conflicts = [
            {
                "id": event["id"],
                "calendar_id": event["calendar_id"],
                "person_id": event["person_id"],
                "title": event["title"],
                "start": event["start"],
                "end": event["end"],
                "timezone": event["timezone"],
                "all_day": event["all_day"],
            }
            for event in listing["events"]
            if event["busy"] and event["status"] != "cancelled"
        ]
        checked = bool(listing["complete"])
        return {
            "start": listing["start"],
            "end": listing["end"],
            "timezone": listing["timezone"],
            "calendars": listing["calendars"],
            "checked": checked,
            "available": checked and not conflicts,
            "conflicts": conflicts,
            "unavailable_calendars": listing["unavailable_calendars"],
            "truncated_calendars": listing["truncated_calendars"],
        }

    def _authorized_calendars(
        self,
        caller_entity_id: str | None,
        *,
        calendar_id: str | None,
        person_id: str | None,
    ) -> tuple[CalendarBinding, ...]:
        if caller_entity_id is None:
            raise CalendarAuthorizationError(
                "Calendar access requires an authenticated household identity"
            )
        accessible = [
            binding
            for binding in self._bindings
            if binding.readable_by(caller_entity_id)
        ]
        if calendar_id is not None:
            accessible = [
                binding for binding in accessible if binding.id == calendar_id
            ]
        if person_id is not None:
            accessible = [
                binding for binding in accessible if binding.person_id == person_id
            ]
        if not accessible:
            raise CalendarAuthorizationError(
                "The requested calendar is not authorized for the caller"
            )
        return tuple(accessible)

    def _parse_window(self, start: str, end: str) -> tuple[datetime, datetime]:
        window_start = parse_calendar_datetime(start, self._default_timezone)
        window_end = parse_calendar_datetime(end, self._default_timezone)
        if window_end <= window_start and _is_date_only(end.strip()):
            window_end = window_end + timedelta(days=1)
        if window_end <= window_start:
            raise ValueError("end must be after start")
        return window_start, window_end


def parse_calendar_datetime(value: str, default_timezone: ZoneInfo) -> datetime:
    """Parse a date or ISO datetime, preserving timezone when supplied."""
    text = value.strip()
    if not text:
        raise ValueError("Date-time value cannot be empty")
    if _is_date_only(text):
        parsed_date = date.fromisoformat(text)
        return datetime(
            parsed_date.year,
            parsed_date.month,
            parsed_date.day,
            tzinfo=default_timezone,
        )
    normalized = text[:-1] + "+00:00" if text.endswith("Z") else text
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as error:
        raise ValueError("Date-time value must be an ISO 8601 date or datetime") from error
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=default_timezone)
    return parsed


def _is_date_only(value: str) -> bool:
    try:
        date.fromisoformat(value)
    except ValueError:
        return False
    return "T" not in value and " " not in value


def _isoformat(value: datetime) -> str:
    return value.isoformat()


def _aware(value: datetime, default_timezone: ZoneInfo) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=default_timezone)
    return value


def _cortex_event(binding: CalendarBinding, event: ProviderEvent) -> dict[str, Any]:
    return {
        "id": f"{binding.id}:{event.provider_event_id}",
        "calendar_id": binding.id,
        "person_id": binding.person_id,
        "title": event.title,
        "start": _isoformat(event.start),
        "end": _isoformat(event.end),
        "timezone": event.timezone,
        "all_day": event.all_day,
        "status": event.status,
        "location": event.location,
        "busy": event.busy,
    }


class GoogleCalendarProvider:
    """Google Calendar Events API adapter. Credentials never leave this type."""

    def __init__(
        self,
        *,
        client_id: str,
        client_secret: str,
        refresh_token: str,
        default_timezone: str,
        http_json: Any | None = None,
        fetch_budget_seconds: float = GOOGLE_FETCH_BUDGET_SECONDS,
    ) -> None:
        if fetch_budget_seconds <= 0:
            raise ValueError("fetch_budget_seconds must be greater than zero")
        self._client_id = client_id
        self._client_secret = client_secret
        self._refresh_token = refresh_token
        self._default_timezone = default_timezone
        self._http_json = http_json or request_json
        self._fetch_budget_seconds = fetch_budget_seconds
        self._token: str | None = None
        self._token_expires_at = 0.0
        self._lock = asyncio.Lock()

    def __repr__(self) -> str:
        return "GoogleCalendarProvider(credentials=redacted)"

    async def list_events(
        self,
        provider_calendar_id: str,
        start: datetime,
        end: datetime,
        *,
        limit: int,
    ) -> ProviderEventBatch:
        deadline = monotonic() + self._fetch_budget_seconds
        events: list[ProviderEvent] = []
        page_token: str | None = None
        seen_page_tokens: set[str] = set()
        while len(events) < limit:
            try:
                token = await self._access_token(deadline=deadline)
                payload = await self._get_events(
                    provider_calendar_id,
                    start,
                    end,
                    limit=limit - len(events),
                    token=token,
                    page_token=page_token,
                    deadline=deadline,
                )
            except _CalendarDeadlineExceeded:
                return ProviderEventBatch(tuple(events), complete=False)
            except CalendarUnavailableError:
                if events:
                    return ProviderEventBatch(tuple(events), complete=False)
                raise
            items = payload.get("items")
            if isinstance(items, list):
                for item in items:
                    if not isinstance(item, Mapping):
                        continue
                    normalized = normalize_google_event(
                        item,
                        self._default_timezone,
                    )
                    if normalized is not None:
                        events.append(normalized)
                        if len(events) >= limit:
                            break

            next_page_token = payload.get("nextPageToken")
            if not isinstance(next_page_token, str) or not next_page_token:
                return ProviderEventBatch(tuple(events), complete=True)
            if next_page_token in seen_page_tokens:
                raise CalendarUnavailableError(
                    "The calendar service is temporarily unavailable"
                )
            if len(events) >= limit:
                return ProviderEventBatch(tuple(events), complete=False)
            seen_page_tokens.add(next_page_token)
            page_token = next_page_token

        return ProviderEventBatch(tuple(events), complete=True)

    async def _get_events(
        self,
        provider_calendar_id: str,
        start: datetime,
        end: datetime,
        *,
        limit: int,
        token: str,
        page_token: str | None = None,
        retry_unauthorized: bool = True,
        deadline: float,
    ) -> Mapping[str, Any]:
        query_parameters = {
            "timeMin": _rfc3339(start),
            "timeMax": _rfc3339(end),
            "singleEvents": "true",
            "orderBy": "startTime",
            "maxResults": str(min(max(limit, 1), MAX_EVENT_LIMIT)),
            "timeZone": self._default_timezone,
        }
        if page_token is not None:
            query_parameters["pageToken"] = page_token
        query = urlencode(query_parameters)
        url = (
            GOOGLE_EVENTS_URL.format(
                calendar_id=quote(provider_calendar_id, safe="")
            )
            + "?"
            + query
        )
        self._require_remaining(deadline, HTTP_TIMEOUT_SECONDS)
        status, payload = await self._request(
            "GET",
            url,
            {"Authorization": f"Bearer {token}", "Accept": "application/json"},
            None,
            deadline=deadline,
        )
        if status == 401 and retry_unauthorized:
            self._require_remaining(deadline, HTTP_TIMEOUT_SECONDS * 2)
            token = await self._access_token(
                force_refresh=True,
                deadline=deadline,
            )
            return await self._get_events(
                provider_calendar_id,
                start,
                end,
                limit=limit,
                token=token,
                page_token=page_token,
                retry_unauthorized=False,
                deadline=deadline,
            )
        if status in {403, 404}:
            raise CalendarUnavailableError(
                "The calendar service is temporarily unavailable"
            )
        if status != 200 or not isinstance(payload, Mapping):
            raise CalendarUnavailableError(
                "The calendar service is temporarily unavailable"
            )
        return payload

    async def _access_token(
        self,
        *,
        force_refresh: bool = False,
        deadline: float,
    ) -> str:
        remaining = self._remaining(deadline)
        try:
            await asyncio.wait_for(self._lock.acquire(), timeout=remaining)
        except TimeoutError as error:
            raise _CalendarDeadlineExceeded from error
        try:
            if (
                not force_refresh
                and self._token is not None
                and monotonic() < self._token_expires_at
            ):
                return self._token
            self._require_remaining(deadline, HTTP_TIMEOUT_SECONDS)
            body = urlencode(
                {
                    "client_id": self._client_id,
                    "client_secret": self._client_secret,
                    "refresh_token": self._refresh_token,
                    "grant_type": "refresh_token",
                }
            ).encode("utf-8")
            status, payload = await self._request(
                "POST",
                GOOGLE_TOKEN_URL,
                {
                    "Accept": "application/json",
                    "Content-Type": "application/x-www-form-urlencoded",
                },
                body,
                deadline=deadline,
            )
            token = (
                payload.get("access_token")
                if status == 200 and isinstance(payload, Mapping)
                else None
            )
            expires_in = (
                payload.get("expires_in")
                if isinstance(payload, Mapping)
                else None
            )
            if not isinstance(token, str) or not token:
                raise CalendarUnavailableError(
                    "The calendar service is temporarily unavailable"
                )
            lifetime = (
                float(expires_in)
                if isinstance(expires_in, (int, float)) and not isinstance(expires_in, bool)
                else 3600.0
            )
            self._token = token
            self._token_expires_at = monotonic() + max(
                lifetime - TOKEN_EXPIRY_SKEW_SECONDS,
                30.0,
            )
            return token
        finally:
            self._lock.release()

    async def _request(
        self,
        method: str,
        url: str,
        headers: Mapping[str, str],
        data: bytes | None,
        *,
        deadline: float,
    ) -> tuple[int, Any]:
        remaining = self._remaining(deadline)
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(
                    self._http_json,
                    method,
                    url,
                    headers,
                    data,
                ),
                timeout=remaining,
            )
        except TimeoutError as error:
            raise _CalendarDeadlineExceeded from error

    @staticmethod
    def _remaining(deadline: float) -> float:
        remaining = deadline - monotonic()
        if remaining <= 0:
            raise _CalendarDeadlineExceeded
        return remaining

    @staticmethod
    def _require_remaining(deadline: float, required: float) -> None:
        if deadline - monotonic() < required:
            raise _CalendarDeadlineExceeded


def normalize_google_event(
    item: Mapping[str, Any],
    default_timezone: str,
) -> ProviderEvent | None:
    event_id = item.get("id")
    if not isinstance(event_id, str) or not event_id:
        return None
    status = item.get("status")
    if status == "cancelled":
        return None
    status_name = status if isinstance(status, str) and status else "confirmed"
    title = item.get("summary")
    title_text = title.strip() if isinstance(title, str) and title.strip() else "(untitled)"
    start = _google_boundary(item.get("start"), default_timezone)
    end = _google_boundary(item.get("end"), default_timezone)
    if start is None or end is None:
        return None
    start_at, start_tz, start_all_day = start
    end_at, end_tz, end_all_day = end
    location = item.get("location")
    location_text = location.strip() if isinstance(location, str) and location.strip() else None
    transparency = item.get("transparency")
    busy = transparency != "transparent"
    return ProviderEvent(
        provider_event_id=event_id,
        title=title_text,
        start=start_at,
        end=end_at,
        timezone=start_tz or end_tz or default_timezone,
        all_day=start_all_day or end_all_day,
        status=status_name,
        location=location_text,
        busy=busy,
    )


def _google_boundary(
    payload: Any,
    default_timezone: str,
) -> tuple[datetime, str, bool] | None:
    if not isinstance(payload, Mapping):
        return None
    timezone_name = payload.get("timeZone")
    zone_name = (
        timezone_name
        if isinstance(timezone_name, str) and timezone_name
        else default_timezone
    )
    date_time = payload.get("dateTime")
    if isinstance(date_time, str) and date_time:
        normalized = date_time[:-1] + "+00:00" if date_time.endswith("Z") else date_time
        try:
            parsed = datetime.fromisoformat(normalized)
        except ValueError:
            return None
        if parsed.tzinfo is None:
            try:
                parsed = parsed.replace(tzinfo=ZoneInfo(zone_name))
            except ZoneInfoNotFoundError:
                parsed = parsed.replace(tzinfo=ZoneInfo(default_timezone))
                zone_name = default_timezone
        return parsed, zone_name, False
    date_value = payload.get("date")
    if isinstance(date_value, str) and date_value:
        try:
            parsed_date = date.fromisoformat(date_value)
            zone = ZoneInfo(zone_name)
        except (ValueError, ZoneInfoNotFoundError):
            return None
        return (
            datetime(
                parsed_date.year,
                parsed_date.month,
                parsed_date.day,
                tzinfo=zone,
            ),
            zone_name,
            True,
        )
    return None


def _rfc3339(value: datetime) -> str:
    aware = value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
    return aware.isoformat()


def request_json(
    method: str,
    url: str,
    headers: Mapping[str, str],
    data: bytes | None,
) -> tuple[int, Any]:
    request = Request(url, data=data, headers=dict(headers), method=method)
    try:
        with urlopen(request, timeout=HTTP_TIMEOUT_SECONDS) as response:
            status = int(response.status)
            raw = response.read()
    except HTTPError as error:
        status = int(error.code)
        try:
            raw = error.read()
        except Exception:
            raw = b""
    except (URLError, TimeoutError, OSError) as error:
        raise CalendarUnavailableError(
            "The calendar service is temporarily unavailable"
        ) from error
    if not raw:
        return status, None
    try:
        return status, json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CalendarUnavailableError(
            "The calendar service is temporarily unavailable"
        ) from error


def _secret_value(value: Any) -> str | None:
    if value is None:
        return None
    getter = getattr(value, "get_secret_value", None)
    if callable(getter):
        extracted = getter()
        return extracted if isinstance(extracted, str) and extracted else None
    if isinstance(value, str) and value:
        return value
    return None


def calendar_service_from_settings(settings: Any) -> CalendarService:
    bindings = tuple(
        CalendarBinding(
            id=binding.id,
            person_id=binding.person_id,
            provider_calendar_id=binding.provider_calendar_id,
            readers=tuple(binding.readers),
        )
        for binding in getattr(settings, "calendar_bindings", ())
    )
    client_id = getattr(settings, "google_calendar_client_id", None)
    client_secret = _secret_value(
        getattr(settings, "google_calendar_client_secret", None)
    )
    refresh_token = _secret_value(
        getattr(settings, "google_calendar_refresh_token", None)
    )
    default_timezone = getattr(settings, "calendar_timezone", "America/Los_Angeles")
    if client_id and client_secret and refresh_token:
        provider: CalendarProvider = GoogleCalendarProvider(
            client_id=client_id,
            client_secret=client_secret,
            refresh_token=refresh_token,
            default_timezone=default_timezone,
        )
    else:
        provider = UnconfiguredCalendarProvider()
    return CalendarService(
        bindings,
        provider,
        default_timezone=default_timezone,
    )
