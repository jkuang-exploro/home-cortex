"""Trusted request context for semantic household facts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class SpeakerContext:
    """Trusted household perspective used to resolve symbolic ``self``."""

    speaker_id: str | None
    household_id: str | None
    locale: str | None
    timezone: str | None


@dataclass(frozen=True)
class AgentRequestContext:
    """Trusted identities and clock used to resolve semantic references."""

    caller_entity_id: str | None
    assistant_id: str
    assistant_display_name: str
    household_id: str | None
    current_time: datetime
    locale: str | None = None

    @property
    def speaker(self) -> SpeakerContext:
        return SpeakerContext(
            speaker_id=self.caller_entity_id,
            household_id=self.household_id,
            locale=self.locale,
            timezone=(
                str(self.current_time.tzinfo)
                if self.current_time.tzinfo is not None
                else None
            ),
        )
