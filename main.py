from agent import agent
from config import settings

# False = short one-line trace (tool names + brief args);
# True = full arguments and longer tool result previews (for debugging)
VERBOSE = False


def message_text(msg) -> str:
    """Extract plain text from a model message.

    Claude's content can be a plain string or a list of blocks
    (thinking / tool_use / text) — we only want the text blocks.
    """
    content = msg.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        return "\n".join(p for p in parts if p)
    return ""


def main():
    print("Research Agent (type 'exit' to quit)")
    print("-" * 40)

    # one session = one conversation thread (memory is stored under this id);
    # recursion_limit caps the number of agent steps so it can't loop forever
    config = {
        "configurable": {"thread_id": "main"},
        "recursion_limit": settings.max_iterations,
    }

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        print("\n🔍 Researching... (this may take a minute)")

        try:
            for chunk in agent.stream(
                {"messages": [("user", user_input)]},
                config,
            ):
                # events from the model: either tool calls or a text answer
                # (the node is called "model" in langchain >= 1.x, "agent" in older versions)
                model_node = chunk.get("model") or chunk.get("agent")
                if model_node and "messages" in model_node:
                    for msg in model_node["messages"]:
                        if getattr(msg, "tool_calls", None):
                            for tc in msg.tool_calls:
                                if VERBOSE:
                                    print(f"\n🧠 AI calls: {tc['name']}({tc['args']})")
                                else:
                                    args = str(tc["args"])[:60]
                                    print(f"🧠 AI calls: {tc['name']}({args}...)")
                        text = message_text(msg)
                        if text:
                            print(f"\n🤖 Agent: {text}")
                # events from tools: confirm the call worked (short by default)
                if "tools" in chunk and "messages" in chunk["tools"]:
                    for msg in chunk["tools"]["messages"]:
                        if VERBOSE:
                            preview = str(msg.content)[:200]
                            print(f"🔧 Tool [{msg.name}]: {preview}...")
                        else:
                            print(f"🔧 Tool [{msg.name}]: done ({len(str(msg.content))} chars)")
        except KeyboardInterrupt:
            print("\n⏹  Research interrupted.")
        except Exception as e:
            print(f"\n⚠️  Agent stopped: {e}")


if __name__ == "__main__":
    main()
