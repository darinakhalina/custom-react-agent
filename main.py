from agent import run_agent


def main():
    print("Research Agent (type 'exit' to quit)")
    print("-" * 40)

    # dialog memory: one list for the whole session — every question,
    # tool call and answer is appended here, so follow-ups have context
    conversation = []

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

        print("\n🔍 Researching... (this may take a minute)\n")

        # remember the clean history length so we can roll back if the run
        # is interrupted mid-way (a tool_use without its tool_result would
        # make the history invalid for the next API call)
        checkpoint = len(conversation)
        conversation.append({"role": "user", "content": user_input})

        try:
            answer = run_agent(conversation)
        except KeyboardInterrupt:
            del conversation[checkpoint:]  # roll back to the clean state
            print("\n⏹  Research interrupted.")
            continue
        except Exception as e:
            del conversation[checkpoint:]
            print(f"\n⚠️  Agent stopped: {e}")
            continue

        # the Anthropic API rejects messages with empty content, so a truncated
        # (e.g. max_tokens) response with no text must not be stored as-is —
        # that would permanently break every future call in this session
        if not answer:
            answer = "(agent stopped without a final answer — try rephrasing or asking again)"

        # keep the model's final answer in the history too,
        # otherwise it won't remember what it replied
        conversation.append({"role": "assistant", "content": answer})

        print(f"\n🤖 Agent: {answer}")


if __name__ == "__main__":
    main()
