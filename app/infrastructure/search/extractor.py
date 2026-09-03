# Refs (read these first, in this order, to understand the module as a whole):
# - Real Python, "Async IO in Python": https://realpython.com/async-io-python/
#   (background: why fetch is native async but extraction runs in a thread)
# - asyncio task/coroutine reference: https://docs.python.org/3/library/asyncio-task.html
# - trafilatura docs: https://trafilatura.readthedocs.io/en/latest/
# - trafilatura usage from Python: https://trafilatura.readthedocs.io/en/latest/usage-python.html
# - httpx: https://www.python-httpx.org/
# - cachetools: https://cachetools.readthedocs.io/en/stable/
#
# Tavily's snippet (SearchResult.snippet) is a short excerpt, not the full
# article. This module fetches the actual page and extracts clean article
# text from the surrounding HTML noise (nav bars, ads, footers) — this is
# what actually gets chunked and embedded for retrieval, not the snippet.

import asyncio
import hashlib
import logging

import httpx
import trafilatura
from cachetools import TTLCache

logger = logging.getLogger(__name__)

# Refs:
# - httpx timeouts (why an explicit timeout, and what the default is if
#   you don't set one): https://www.python-httpx.org/advanced/timeouts/
# - httpx exception hierarchy (TimeoutException, ConnectError, HTTPStatusError,
#   all under HTTPError): https://www.python-httpx.org/exceptions/
#
# 10s is generous enough for slow-but-legitimate pages while still
# bounding the worst case; the retrieval pipeline already tolerates any
# single source failing (see service.py's _run_search_tool), so a
# timeout here just turns "hangs forever" into "skipped, like any other
# bad source."
FETCH_TIMEOUT_SECONDS = 10.0

# Ref: HTTP User-Agent header semantics — https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/User-Agent
# Some sites reject requests with no/blank User-Agent outright.
_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; RagnarResearchBot/1.0)"}

# Refs:
# - cachetools.TTLCache: https://cachetools.readthedocs.io/en/stable/#cachetools.TTLCache
# - Why no lock is needed around a plain dict-like cache accessed only
#   from the event loop thread: https://docs.python.org/3/library/asyncio-dev.html#concurrency-and-multithreading
#   (asyncio coroutines are cooperatively scheduled; a mutation with no
#   `await` inside it can't be interleaved with another coroutine's)
#
# Page content for a given URL rarely changes within the lifetime of a
# few requests, and re-fetching + re-extracting the same page for every
# question that happens to search for it wastes both the round trip and
# the CPU time trafilatura spends parsing the HTML. 30 minutes balances
# "avoid redundant work within a burst of similar questions" against
# "don't serve stale content for too long."
_CACHE_TTL_SECONDS = 1800
_extraction_cache: TTLCache = TTLCache(maxsize=512, ttl=_CACHE_TTL_SECONDS)

# Sentinel distinguishing "cached: this URL had nothing extractable" from
# "not cached yet" — cachetools' .get() returns None for a genuine miss,
# but None is also the value we'd otherwise want to cache for a bad URL.
# Ref (sentinel object pattern): https://docs.python.org/3/library/hashlib.html#the-sentinel-pattern
# (general Python idiom; not hashlib-specific — see also PEP 661 for a
# proposed stdlib sentinel: https://peps.python.org/pep-0661/)
_NO_CONTENT = object()


def _cache_key(url: str) -> str:
    # Ref: hashlib docs — https://docs.python.org/3/library/hashlib.html
    # URLs can be long; hashing keeps cache keys a fixed, small size.
    return hashlib.sha256(url.encode("utf-8")).hexdigest()


async def _fetch_html(url: str) -> str | None:
    # Refs:
    # - httpx.AsyncClient: https://www.python-httpx.org/async/
    # - follow_redirects: https://www.python-httpx.org/quickstart/#redirection-and-history
    # - response.raise_for_status(): https://www.python-httpx.org/quickstart/#errors
    try:
        async with httpx.AsyncClient(
            timeout=FETCH_TIMEOUT_SECONDS,
            follow_redirects=True,
            headers=_HEADERS,
        ) as client:
            response = await client.get(url)
            response.raise_for_status()
            return response.text
    except httpx.HTTPError as exc:
        # httpx.HTTPError is the common base for both request-level
        # failures (timeout, connection refused — httpx.RequestError) and
        # response-level failures (4xx/5xx via raise_for_status —
        # httpx.HTTPStatusError). Ref: https://www.python-httpx.org/exceptions/
        logger.warning(f"Could not fetch page for extraction: {url} ({exc})")
        return None


def _extract_sync(html: str, url: str) -> str | None:
    # Ref: trafilatura.extract() parameters — https://trafilatura.readthedocs.io/en/latest/usage-python.html#extraction
    text = trafilatura.extract(html, url=url)
    if not text or not text.strip():
        logger.warning(f"No extractable content found at: {url}")
        return None
    return text


async def extract_page_text(url: str) -> str | None:
    """
    Fetch a page and extract its main article text, off the event loop
    for the extraction step (the fetch itself is native async via httpx).

    Ref (asyncio.to_thread — why the CPU-bound trafilatura.extract call is
         offloaded, but the httpx fetch is not):
        https://docs.python.org/3/library/asyncio-task.html#asyncio.to_thread

    Returns None if the page couldn't be fetched within
    FETCH_TIMEOUT_SECONDS or had no extractable content (paywall, JS-only
                                                         page, non-HTML resource), the caller is expected to skip that source
    rather than fail the whole request over one bad page.

    Results (including "nothing extractable") are cached per URL for
    _CACHE_TTL_SECONDS, so repeated requests for the same page within
    that window skip the fetch and extraction entirely.
    """
    key = _cache_key(url)

    cached = _extraction_cache.get(key)
    if cached is not None:
        return None if cached is _NO_CONTENT else cached

    html = await _fetch_html(url)
    if html is None:
        return None

    text = await asyncio.to_thread(_extract_sync, html, url)
    _extraction_cache[key] = text if text is not None else _NO_CONTENT
    return text
