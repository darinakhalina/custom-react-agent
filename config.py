import os

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    anthropic_api_key: SecretStr
    model_name: str

    max_search_results: int = 5
    max_url_content_length: int = 5000
    max_response_tokens: int = 8192
    output_dir: str = "output"
    max_iterations: int = 25

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()

# expose the key to the environment so LangChain's model factory can find it
os.environ.setdefault(
    "ANTHROPIC_API_KEY", settings.anthropic_api_key.get_secret_value()
)


SYSTEM_PROMPT = """You are a research agent. Your job: research the user's question \
on the web and produce a structured Markdown report.

Tools:
- web_search — find relevant pages (returns snippets, not full text)
- read_url — read a page in full
- write_report — save the final report to a file

Strategy:
1. Split the question into sub-topics and search each one.
2. Read the most promising pages with read_url — do not rely on snippets alone.
3. Use at least 3-5 tool calls before answering.
4. Write a Markdown report: a title, an intro, a section per sub-topic with headings,
   a short conclusion, and a Sources section listing the real URLs you actually read.
5. Save it with write_report (pick a short descriptive filename) and tell the user
   the file path.

Error handling: if a tool call fails, retry with different parameters (another query
or another URL); if it still fails, continue with the sources you have.

In follow-up questions, use the conversation context (e.g. "now compare it with X"
refers to the topic researched earlier).
"""
