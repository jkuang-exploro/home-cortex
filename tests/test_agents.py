import pytest

from home_cortex.agents import UnknownAgentError, get_agent, list_agents


def test_steward_agent_loads_successfully() -> None:
    steward = get_agent("steward")
    prompt = " ".join(steward.prompt.split())

    assert steward.id == "steward"
    assert steward.model.provider == "ollama"
    assert "source of truth" in prompt
    assert "continue using tools" in prompt
    assert "`spouse_of` is symmetric" in prompt
    assert "lives_in.start" in prompt
    assert "household-fact answer must be supported" in prompt
    assert "meaning of the request" in prompt
    assert "memorized sentence" in prompt
    assert '"我女儿是谁"' not in prompt
    assert '"Who is my daughter?"' not in prompt
    assert "never a wedding or anniversary date" in prompt
    assert "`person.dob`" in prompt
    assert "call `get_entity`" in prompt
    assert "household roster semantics" in prompt.casefold()
    assert "Casual conversation" in prompt
    assert "does not by itself request graph data" in prompt
    assert "graph and calendar tools are read-only" in prompt
    assert "Use `calculate` for exact arithmetic" in prompt
    assert "`calendar.list_events`" in prompt
    assert "`calendar.check_availability`" in prompt
    assert "If `complete` is false" in prompt
    assert "`unavailable_calendars`" in prompt
    assert "trusted household clock" in prompt
    assert "Do not append a service slogan" in prompt
    assert "location:fort_cerritos" in prompt
    assert "item:fort_cerritos_house" in prompt
    assert "`hosted_by` is directed from Space" in prompt
    assert "`hosts_space` is" in prompt
    assert "Never infer a hosted space" in prompt
    assert "Item-container lookup" in prompt
    assert "`entity_type` set to `item`" in prompt
    assert "For each hosted Space, traverse `located_in`" in prompt
    assert "Home-space lookup" in prompt
    assert "`contained_in`" not in prompt
    assert "`part_of`" not in prompt
    assert "People live at a location, never at a space" in prompt


def test_steward_has_localized_identity_and_home_scope() -> None:
    steward = get_agent("steward")

    assert steward.display_name == "老管家"
    assert steward.settings["home_entity_id"] == "location:fort_cerritos"
    assert steward.settings["localized_identity"] == {
        "en": "the butler",
        "zh": "老管家",
    }
    assert steward.settings["reception"]["greetings"]["zh"]["guest"].startswith(
        "{address_as}，您好"
    )
    assert 'In English, refer to yourself as "the butler"' in steward.prompt
    assert 'In Chinese, refer to yourself as "老管家"' in steward.prompt
    assert "never substitute a generic label" in steward.prompt
    assert 'Never use "老管家" or' in steward.prompt
    assert '"the butler" as a salutation for the user' in steward.prompt
    assert "Fort Cerritos" in steward.prompt
    assert "喜瑞匡家" in steward.prompt


def test_unknown_agent_id_fails_cleanly() -> None:
    with pytest.raises(UnknownAgentError, match="Unknown agent 'accountant'"):
        get_agent("accountant")


def test_steward_receives_only_its_configured_tools() -> None:
    steward = get_agent("steward")
    definition_names = tuple(
        tool["function"]["name"] for tool in steward.tool_definitions
    )

    assert steward.allowed_tools == (
        "get_entity",
        "search_entities",
        "get_relationships",
        "calculate",
        "calendar.list_events",
        "calendar.check_availability",
    )
    assert definition_names == steward.allowed_tools
    assert [agent.id for agent in list_agents()] == ["steward"]
