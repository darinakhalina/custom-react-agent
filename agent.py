from anthropic import Anthropic

from config import settings, SYSTEM_PROMPT
from tools import TOOL_SCHEMAS

# direct API client — the "phone line" to Claude (no framework);
# the API key is read from the environment (config.py puts it there)
client = Anthropic()


def call_model(conversation: list):
    """One call to Claude: send the whole conversation + the tool menu.

    The API is stateless — it remembers nothing between calls — so we
    pass the full message history every time. The reply is either a text
    answer or a request to call one of our tools.
    """
    return client.messages.create(
        model=settings.model_name,
        max_tokens=8192,
        system=SYSTEM_PROMPT,
        messages=conversation,
        tools=TOOL_SCHEMAS,
    )
