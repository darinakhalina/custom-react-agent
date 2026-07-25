from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

from config import settings, SYSTEM_PROMPT
from tools import web_search, read_url, write_report

# model id from settings (.env), e.g. "anthropic:claude-sonnet-5"
llm = f"anthropic:{settings.model_name}"

# @tool-decorated functions the agent is allowed to call
tools = [web_search, read_url, write_report]

# conversation memory (checkpointer): keeps message history within a session,
# so follow-up questions like "now compare it with X" have context
memory = InMemorySaver()

# the agent itself: model + tools + memory + instructions (system prompt)
agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=SYSTEM_PROMPT,
    checkpointer=memory,
)
