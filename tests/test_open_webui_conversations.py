"""Native conversation create-then-message flow replaces the Open WebUI hook."""

from types import SimpleNamespace

from fastapi.testclient import TestClient

from home_cortex.api import VIRTUAL_MODEL, app
from home_cortex.conversations import ConversationStore
from home_cortex.greetings import GreetingService

from test_api import FakeAgent, FakeHealthDatabase, FakeIdentityRetrieval


def test_first_turn_is_scoped_to_the_created_conversation() -> None:
    previous = {
        "agent": getattr(app.state, "agent", None),
        "agents": getattr(app.state, "agents", None),
        "settings": getattr(app.state, "settings", None),
        "retrieval": getattr(app.state, "retrieval", None),
        "greetings": getattr(app.state, "greetings", None),
        "conversations": getattr(app.state, "conversations", None),
        "database": getattr(app.state, "database", None),
    }
    agent = FakeAgent()
    retrieval = FakeIdentityRetrieval()
    app.state.agent = agent
    app.state.agents = {"steward": agent}
    app.state.settings = SimpleNamespace(cortex_api_key=None, cortex_identity_map={})
    app.state.retrieval = retrieval
    app.state.greetings = GreetingService(retrieval)
    app.state.conversations = ConversationStore()
    app.state.database = FakeHealthDatabase()
    client = TestClient(app, raise_server_exceptions=True)
    try:
        created = client.post(
            "/conversations",
            json={"language": "en", "model": VIRTUAL_MODEL},
        )
        conversation_id = created.json()["id"]
        first = client.post(
            f"/conversations/{conversation_id}/messages",
            json={"content": "How old is son1?", "stream": False},
        )
        follow = client.post(
            f"/conversations/{conversation_id}/messages",
            json={"content": "When is his tenth birthday?", "stream": False},
        )
        other = client.post(
            "/conversations",
            json={"language": "en", "model": VIRTUAL_MODEL},
        )
        assert first.status_code == 200
        assert follow.status_code == 200
        assert other.json()["id"] != conversation_id
        loaded = client.get(f"/conversations/{conversation_id}")
        assert [item["content"] for item in loaded.json()["messages"] if item["role"] == "user"] == [
            "How old is son1?",
            "When is his tenth birthday?",
        ]
    finally:
        client.close()
        for name, value in previous.items():
            if value is None:
                if hasattr(app.state, name):
                    delattr(app.state, name)
            else:
                setattr(app.state, name, value)
