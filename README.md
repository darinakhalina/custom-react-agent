# Custom ReAct Agent

A research agent with a **hand-written ReAct loop** — no `create_react_agent`,
no framework agent abstractions. You ask a question in the terminal; the agent
searches the web, reads relevant pages, and writes a structured Markdown
report, deciding which tools to call on its own.

Evolution of [research-agent](https://github.com/darinakhalina/research-agent):
the LangChain agent is replaced with a custom loop built directly on the
Anthropic API — tools as JSON Schema, manual message history, manual tool
dispatch.

## Tools

- `web_search` — DuckDuckGo search (no API key)
- `read_url` — fetch full page text, trimmed to fit the context window
- `write_report` — save the final Markdown report to a file

The agent keeps conversation memory within a session (follow-up questions work)
and has a step limit so it can't loop forever.

## Setup

```bash
git clone https://github.com/darinakhalina/custom-react-agent.git
cd custom-react-agent
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then add your key
```

Fill in `.env`:

```
ANTHROPIC_API_KEY=sk-ant-your-api-key-here
MODEL_NAME=claude-sonnet-5
```

Get a key at [console.anthropic.com](https://console.anthropic.com). `.env` is
git-ignored — never commit it.

## Usage

```bash
python main.py
```

Reports are saved to the `output/` directory. A sample generated report lives
in `example_output/report.md`.

## Project structure

| File | Role |
|------|------|
| `main.py` | Interactive terminal loop |
| `agent.py` | Custom ReAct loop: send → parse tool calls → execute → repeat |
| `tools.py` | Tool implementations + JSON Schema definitions |
| `config.py` | Settings + system prompt |
