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


def response_text(response) -> str:
    """Collect the text blocks from a model response into one string."""
    return "\n".join(
        block.text for block in response.content if block.type == "text"
    )


def run_agent(conversation: list) -> str:
    """The ReAct loop: call the model, execute requested tools, repeat.

    Stops when the model returns a final text answer (no tool_use)
    or when the iteration limit is reached.
    """
    for _ in range(settings.max_iterations):
        response = call_model(conversation)

        # no tool requests -> the model is done, return its final text
        if response.stop_reason != "tool_use":
            return response_text(response)

        # TODO (next step): execute the requested tools via TOOL_REGISTRY
        # and append the results to the conversation

    return "Stopped: reached the iteration limit without a final answer."
