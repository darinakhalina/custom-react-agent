from pathlib import Path

from ddgs import DDGS
import trafilatura
from trafilatura.settings import use_config

from config import settings

# trafilatura has no timeout by default, which lets a hanging page stall
# the whole agent loop — cap it explicitly
_fetch_config = use_config()
_fetch_config.set("DEFAULT", "DOWNLOAD_TIMEOUT", "10")


def write_report(filename: str, content: str) -> str:
    """Save a Markdown report to the output directory. Returns the file path.

    Raises on failure (bad filename, disk error) — the caller (run_agent)
    catches it and reports it to the model as a tool error.
    """
    # make sure the output folder exists (create it if not)
    output_dir = Path(settings.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # strip any directory components (e.g. "../../notes.md") so the file
    # can't be written outside output_dir
    filename = Path(filename).name
    # ensure the file ends with .md
    if not filename.endswith(".md"):
        filename += ".md"
    path = output_dir / filename
    # belt-and-suspenders: confirm the resolved path is still inside output_dir
    if output_dir.resolve() not in path.resolve().parents:
        raise ValueError(f"Invalid filename: {filename}")
    # write the report text into the file
    path.write_text(content, encoding="utf-8")
    return f"Report saved to {path}"


write_report_schema = {
    "name": "write_report",
    "description": (
        "Save the final Markdown report to a file in the output directory. "
        "Returns the saved file path."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "filename": {
                "type": "string",
                "description": "Short descriptive file name, e.g. 'rag-comparison.md'",
            },
            "content": {
                "type": "string",
                "description": "The full Markdown report text",
            },
        },
        "required": ["filename", "content"],
    },
}


def web_search(query: str) -> str:
    """Search the web and return top results (title, URL, snippet).

    An empty result set is a legitimate outcome, not a failure, so it's
    returned normally. Real failures (network error, DDG backend down)
    propagate — run_agent catches them and reports a tool error.
    """
    # query DuckDuckGo and take the top 5 results
    # (results = list of dicts with keys title / href / body)
    results = DDGS().text(query, max_results=settings.max_search_results)
    if not results:
        return f"No results found for: {query}"
    formatted = []
    # loop over each result and build a clean line:
    # number, title, URL, snippet
    for i, r in enumerate(results, 1):
        formatted.append(
            f"{i}. {r.get('title', '')}\n   URL: {r.get('href', '')}\n   {r.get('body', '')}"
        )
    return "\n\n".join(formatted)


# JSON Schema for the tool calling API: what @tool used to build from
# the function signature and docstring, we now declare explicitly.
web_search_schema = {
    "name": "web_search",
    "description": (
        "Search the web and return top results (title, URL, snippet). "
        "Snippets are short, not full pages — use read_url to read a page in full."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The search query"},
        },
        "required": ["query"],
    },
}


def read_url(url: str) -> str:
    """Fetch the main text of a web page (use after web_search to read it in full).

    Raises on any failure to fetch (bad scheme, unreachable page, no
    extractable text) — run_agent catches it and reports a tool error.
    """
    # the model can pick up a URL from page content it just read (indirect
    # prompt injection) — restrict to http(s) so it can't be pointed at
    # file:// or other local/internal schemes
    if not url.lower().startswith(("http://", "https://")):
        raise ValueError(f"Refusing to fetch non-http(s) URL: {url}")
    # download the page, then extract only the main text (no menus/ads)
    # (explicit timeout so a slow/hanging page can't stall the agent loop)
    downloaded = trafilatura.fetch_url(url, config=_fetch_config)
    if not downloaded:
        raise RuntimeError(f"Could not fetch the page: {url}")
    text = trafilatura.extract(downloaded)
    if not text:
        raise RuntimeError(f"No readable content found at: {url}")
    # context engineering: trim long pages so they don't flood the context
    limit = settings.max_url_content_length
    if len(text) > limit:
        text = text[:limit] + "\n\n[...truncated...]"
    return text


read_url_schema = {
    "name": "read_url",
    "description": (
        "Fetch the main text of a web page. "
        "Use after web_search to read a promising page in full. "
        "Long pages are trimmed to fit the context window."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "The page URL to read"},
        },
        "required": ["url"],
    },
}


# what we hand to the API on every call: all tool definitions
TOOL_SCHEMAS = [web_search_schema, read_url_schema, write_report_schema]

# tool registry: maps a tool name from the model's response
# to the actual Python function our loop should execute
TOOL_REGISTRY = {
    "web_search": web_search,
    "read_url": read_url,
    "write_report": write_report,
}
