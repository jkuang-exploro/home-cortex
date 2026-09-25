"""Local llama-server adapter using the shared OpenAI-compatible transport."""

from .openai_compatible import OpenAICompatibleService


class LlamaCppService(OpenAICompatibleService):
    """A configured local model; the chat caller cannot select a filesystem path."""
