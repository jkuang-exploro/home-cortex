from dataclasses import dataclass, field
from importlib import import_module, resources
from types import MappingProxyType
from typing import Any, Mapping

import yaml

from ..tools import get_tool_definitions


class UnknownAgentError(LookupError):
    """Raised when an agent ID or display name is not registered."""


@dataclass(frozen=True)
class ModelConfiguration:
    provider: str
    name: str | None = None


@dataclass(frozen=True)
class AgentDefinition:
    id: str
    display_name: str
    description: str
    prompt: str
    model: ModelConfiguration
    allowed_tools: tuple[str, ...]
    tool_definitions: tuple[Mapping[str, Any], ...]
    settings: Mapping[str, Any] = field(
        default_factory=lambda: MappingProxyType({})
    )


def _load_agent(agent_id: str) -> AgentDefinition:
    package_name = f"{__package__}.{agent_id}"
    package_files = resources.files(package_name)
    raw = yaml.safe_load(package_files.joinpath("config.yaml").read_text("utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Agent {agent_id!r} config must be a YAML object")

    configured_id = _required_string(raw, "id", agent_id)
    if configured_id != agent_id:
        raise ValueError(
            f"Agent directory {agent_id!r} does not match config ID {configured_id!r}"
        )
    display_name = _required_string(raw, "display_name", agent_id)
    description = _required_string(raw, "description", agent_id)

    model = raw.get("model")
    if not isinstance(model, dict):
        raise ValueError(f"Agent {agent_id!r} requires a model configuration")
    provider = _required_string(model, "provider", agent_id)
    model_name = model.get("name")
    if model_name is not None and (
        not isinstance(model_name, str) or not model_name.strip()
    ):
        raise ValueError(f"Agent {agent_id!r} model name must be a string or null")

    policy_module = import_module(f"{package_name}.tools")
    allowed_tools = getattr(policy_module, "ALLOWED_TOOLS", None)
    if not isinstance(allowed_tools, tuple) or not all(
        isinstance(name, str) and name for name in allowed_tools
    ):
        raise ValueError(f"Agent {agent_id!r} ALLOWED_TOOLS must be a tuple of names")
    tool_definitions = tuple(get_tool_definitions(allowed_tools))

    prompt = package_files.joinpath("prompt.md").read_text("utf-8").strip()
    if not prompt:
        raise ValueError(f"Agent {agent_id!r} prompt cannot be empty")

    settings = raw.get("settings") or {}
    if not isinstance(settings, dict):
        raise ValueError(f"Agent {agent_id!r} settings must be a YAML object")

    return AgentDefinition(
        id=agent_id,
        display_name=display_name,
        description=description,
        prompt=prompt,
        model=ModelConfiguration(provider=provider, name=model_name),
        allowed_tools=allowed_tools,
        tool_definitions=tool_definitions,
        settings=MappingProxyType(dict(settings)),
    )


def _required_string(value: Mapping[str, Any], key: str, agent_id: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item.strip():
        raise ValueError(f"Agent {agent_id!r} requires a non-empty {key!r}")
    return item.strip()


_AGENTS = {definition.id: definition for definition in (_load_agent("steward"),)}
_DISPLAY_NAMES = {
    definition.display_name: definition for definition in _AGENTS.values()
}


def get_agent(agent_id: str) -> AgentDefinition:
    try:
        return _AGENTS[agent_id]
    except KeyError as error:
        raise UnknownAgentError(f"Unknown agent {agent_id!r}") from error


def get_agent_by_display_name(display_name: str) -> AgentDefinition:
    try:
        return _DISPLAY_NAMES[display_name]
    except KeyError as error:
        raise UnknownAgentError(f"Unknown agent model {display_name!r}") from error


def list_agents() -> tuple[AgentDefinition, ...]:
    return tuple(_AGENTS.values())
