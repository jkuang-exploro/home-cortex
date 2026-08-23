from datetime import date
from pathlib import Path

from home_cortex.memorable_dates import MemorableDateRegistry


SCHEMA_PATH = Path(__file__).parents[1] / "schemas" / "memorable_dates.yaml"


def test_registry_maps_date_concepts_to_authoritative_graph_fields() -> None:
    registry = MemorableDateRegistry.from_file(SCHEMA_PATH)

    birthday = registry.match("我岳父生日是哪天？")
    anniversary = registry.match("our wedding anniversary")

    assert birthday is not None
    assert (birthday.source_kind, birthday.source_type, birthday.source_field) == (
        "node",
        "person",
        "dob",
    )
    assert anniversary is not None
    assert (
        anniversary.source_kind,
        anniversary.source_type,
        anniversary.source_field,
    ) == ("edge", "spouse_of", "start")


def test_annual_occurrence_is_generic_for_birthdays_and_anniversaries() -> None:
    registry = MemorableDateRegistry.from_file(SCHEMA_PATH)
    as_of = date(2026, 8, 22)

    birthday = registry.occurrence(
        registry.get("birthday"),
        "1961-10-10",
        as_of=as_of,
    )
    anniversary = registry.occurrence(
        registry.get("wedding_anniversary"),
        "2014-05-04",
        as_of=as_of,
    )

    assert birthday is not None
    assert (birthday.next_occurrence, birthday.days_until) == (
        date(2026, 10, 10),
        49,
    )
    assert anniversary is not None
    assert (anniversary.next_occurrence, anniversary.days_until) == (
        date(2027, 5, 4),
        255,
    )


def test_annual_occurrence_handles_leap_day_without_a_date_kind_branch() -> None:
    registry = MemorableDateRegistry.from_file(SCHEMA_PATH)

    occurrence = registry.occurrence(
        registry.get("birthday"),
        "2000-02-29",
        as_of=date(2026, 3, 1),
    )

    assert occurrence is not None
    assert occurrence.next_occurrence == date(2028, 2, 29)


def test_recurrence_does_not_create_an_occurrence_before_the_stored_date() -> None:
    registry = MemorableDateRegistry.from_file(SCHEMA_PATH)

    occurrence = registry.occurrence(
        registry.get("wedding_anniversary"),
        "2028-05-04",
        as_of=date(2026, 8, 22),
    )

    assert occurrence is not None
    assert occurrence.next_occurrence == date(2028, 5, 4)
