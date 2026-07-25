from pathlib import Path

from ddgs import DDGS
import trafilatura

from config import settings


def write_report(filename: str, content: str) -> str:
    """Save a Markdown report to the output directory. Returns the file path."""
    try:
        # make sure the output folder exists (create it if not)
        output_dir = Path(settings.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        # ensure the file ends with .md
        if not filename.endswith(".md"):
            filename += ".md"
        # write the report text into the file
        path = output_dir / filename
        path.write_text(content, encoding="utf-8")
        return f"Report saved to {path}"
    except Exception as e:
        return f"Failed to write report: {e}"


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
    """Search the web and return top results (title, URL, snippet)."""
    try:
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
                f"{i}. {r['title']}\n   URL: {r['href']}\n   {r['body']}"
            )
        return "\n\n".join(formatted)
    except Exception as e:
        return f"Search failed for '{query}': {e}"


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
    """Fetch the main text of a web page (use after web_search to read it in full)."""
    try:
        # download the page, then extract only the main text (no menus/ads)
        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            return f"Could not fetch the page: {url}"
        text = trafilatura.extract(downloaded)
        if not text:
            return f"No readable content found at: {url}"
        # context engineering: trim long pages so they don't flood the context
        limit = settings.max_url_content_length
        if len(text) > limit:
            text = text[:limit] + "\n\n[...truncated...]"
        return text
    except Exception as e:
        return f"Failed to read '{url}': {e}"


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
