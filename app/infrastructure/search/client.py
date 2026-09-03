# Refs:
# - Tavily API reference: https://docs.tavily.com/documentation/api-reference/endpoint/search
# - tavily-python SDK: https://github.com/tavily-ai/tavily-python
#
# This module is the boundary between our app and the Tavily search API.
# It returns raw search results (title, url, snippet) only. Fetching the
# full page and extracting clean text happens separately in extractor.py,
# since Tavily's snippet is a short excerpt, not the full article.

import logging

from pydantic import BaseModel, ConfigDict, HttpUrl
from tavily import AsyncTavilyClient

from app.infrastructure.search.config import TAVILY_API_KEY

logger = logging.getLogger(__name__)

client = AsyncTavilyClient(api_key=TAVILY_API_KEY)


class SearchResult(BaseModel):
    """One raw result from the search API, before the full page is fetched."""

    model_config = ConfigDict(extra="forbid")

    url: HttpUrl
    title: str
    snippet: str


async def search_web(query: str, max_results: int = 5) -> list[SearchResult]:
    """
    Query Tavily and return the top results.

    max_results=5: high enough to give the retrieval pipeline downstream
    (fetch + chunk + embed + rank) a reasonable pool of sources, low
    enough to keep fetch/extraction cost bounded, since every result
    returned here gets its full page fetched next in extractor.py.

    Ref: https://docs.tavily.com/documentation/api-reference/endpoint/search
    The expected body from Tavily result is:

    {
      "query": "Who is Leo Messi?",
      "answer": "Lionel Messi, born in 1987, is an Argentine footballer widely regarded as one of the greatest players of his generation.",
      "images": [],
      "results": [
        {
          "title": "Lionel Messi Facts | Britannica",
          "url": "https://www.britannica.com/facts/Lionel-Messi",
          "content": "Lionel Messi, an Argentine footballer, is widely regarded as one of the greatest football players of his generation.",
          "score": 0.81025416,
          "raw_content": null,
          "favicon": "https://britannica.com/favicon.png",
          "images": [
            {
              "url": "<string>",
              "description": "<string>"
            }
          ],
          "id": "a3f9c2-04"
        }
      ],
      "response_time": "1.67",
      "auto_parameters": {
        "topic": "general",
        "search_depth": "basic"
      },
      "usage": {
        "credits": 1
      },
      "request_id": "123e4567-e89b-12d3-a456-426614174111"
    }
    """

    response = await client.search(query=query, max_results=max_results)

    results = []
    for item in response.get("results", []):
        try:
            results.append(
                SearchResult(
                    url=item["url"],
                    title=item.get("title", ""),
                    snippet=item.get("content", ""),
                )
            )
        except Exception:
            # A malformed result (e.g. an invalid URL) shouldn't take down
            # the whole search; skip it and keep the rest.
            logger.warning(f"Skipping malformed search result: {item!r}")
            continue

    return results
