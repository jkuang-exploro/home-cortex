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
    assert "graph tools are read-only" in prompt
    assert "location:fort_cerritos" in prompt


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
    )
    assert definition_names == steward.allowed_tools
    assert [agent.id for agent in list_agents()] == ["steward"]
