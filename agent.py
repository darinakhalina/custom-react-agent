from anthropic import Anthropic

from config import settings, SYSTEM_PROMPT
from tools import TOOL_SCHEMAS, TOOL_REGISTRY

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

        # the model's reply (with its tool requests) goes into the history first
        conversation.append({"role": "assistant", "content": response.content})

        # execute every requested tool and collect the results
        results = []
        for block in response.content:
            if block.type != "tool_use":
                continue  # skip thinking/text blocks
            args = ", ".join(f'{k}={str(v)[:60]!r}' for k, v in block.input.items())
            print(f"🔧 Tool call: {block.name}({args})")
            func = TOOL_REGISTRY.get(block.name)  # name from the schema -> our function
            if func is None:
                output = f"Unknown tool: {block.name}"  # typo protection, don't crash
            else:
                output = func(**block.input)  # run the real Python function
            print(f"📎 Result: {output[:120]}...\n")
            results.append({
                "type": "tool_result",
                "tool_use_id": block.id,  # which request this answers
                "content": output,
            })

        # hand the results back to the model on the next loop iteration
        conversation.append({"role": "user", "content": results})

    return "Stopped: reached the iteration limit without a final answer."
