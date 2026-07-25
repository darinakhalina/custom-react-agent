from pathlib import Path

from ddgs import DDGS
from langchain_core.tools import tool
import trafilatura

from config import settings


# @tool marks a function as a TOOL for the agent:
# Claude sees its name and docstring and can call it on its own.
@tool
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


@tool
def web_search(query: str) -> str:
    """Search the web and return top results (title, URL, snippet).

    Snippets are short, not full pages — use read_url to read a page in full.
    """
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


@tool
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
