import pytest

from home_cortex.agents import UnknownAgentError, get_agent, list_agents


def test_steward_agent_loads_successfully() -> None:
    steward = get_agent("steward")
    prompt = " ".join(steward.prompt.split())

    assert steward.id == "steward"
    assert steward.model.provider == "ollama"
    assert "source of truth" in prompt
    assert "schema-aware grounding pipeline" in prompt
    assert "get_entity" not in prompt
    assert "Casual conversation" not in prompt
    assert "A mention of a person" in prompt
    assert "Use `calculate` for exact non-trivial arithmetic" in prompt
    assert "`calendar.list_events`" in prompt
    assert "`calendar.check_availability`" in prompt
    assert "If `complete` is false" in prompt
    assert "unavailable or truncated calendars" in prompt
    assert "trusted household clock" in prompt
    assert "Do not append slogans" in prompt
    assert "spouse_of" not in prompt
    assert "person.dob" not in prompt


def test_steward_has_localized_identity_and_home_scope() -> None:
    steward = get_agent("steward")

    assert steward.display_name == "老管家"
    assert steward.settings["home_entity_id"] == "address:fort_cerritos"
    assert steward.settings["localized_identity"] == {
        "en": "the butler",
        "zh": "老管家",
    }
    assert steward.settings["reception"]["greetings"]["zh"]["guest"].startswith(
        "{address_as}，您好"
    )
    assert 'In English, refer to yourself as "the butler"' in steward.prompt
    assert 'In Chinese, refer to yourself as "老管家"' in " ".join(
        steward.prompt.split()
    )
    assert "Your role name identifies you, never the speaker" in steward.prompt
    assert "configured home" in steward.prompt


def test_unknown_agent_id_fails_cleanly() -> None:
    with pytest.raises(UnknownAgentError, match="Unknown agent 'accountant'"):
        get_agent("accountant")


def test_steward_receives_only_its_configured_tools() -> None:
    steward = get_agent("steward")
    definition_names = tuple(
        tool["function"]["name"] for tool in steward.tool_definitions
    )

    assert steward.allowed_tools == (
        "calculate",
        "calendar.list_events",
        "calendar.check_availability",
    )
    assert definition_names == steward.allowed_tools
    assert [agent.id for agent in list_agents()] == ["steward"]
