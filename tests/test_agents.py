import pytest

from home_cortex.agents import UnknownAgentError, get_agent, list_agents


def test_steward_agent_loads_successfully() -> None:
    steward = get_agent("steward")

    assert steward.id == "steward"
    assert steward.model.provider == "ollama"
    assert "source of truth" in steward.prompt
    assert "continue using tools" in steward.prompt


def test_steward_display_name_is_chinese() -> None:
    assert get_agent("steward").display_name == "老管家"


def test_unknown_agent_id_fails_cleanly() -> None:
    with pytest.raises(UnknownAgentError, match="Unknown agent 'accountant'"):
        get_agent("accountant")


def test_steward_receives_only_its_configured_tools() -> None:
    steward = get_agent("steward")
    definition_names = tuple(
        tool["function"]["name"] for tool in steward.tool_definitions
    )

    assert steward.allowed_tools == (
        "search_entities",
        "get_relationships",
    )
    assert definition_names == steward.allowed_tools
    assert [agent.id for agent in list_agents()] == ["steward"]
