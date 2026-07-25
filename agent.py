from anthropic import Anthropic

from config import settings, SYSTEM_PROMPT
from tools import TOOL_SCHEMAS

# direct API client — the "phone line" to Claude (no framework);
# the API key is read from the environment (config.py puts it there)
client = Anthropic()


def call_model(conversation: list):
    """One API call: full conversation + tool schemas -> text answer or tool_use request."""
    return client.messages.create(
        model=settings.model_name,
        max_tokens=settings.max_response_tokens,
        system=SYSTEM_PROMPT,
        messages=conversation,
        tools=TOOL_SCHEMAS,
    )
