# Refs (read these first, in this order, to understand the module as a whole):
# - pytest-asyncio markers: https://pytest-asyncio.readthedocs.io/en/stable/reference/markers/index.html
# - respx (httpx mocking library used below): https://lundberg.github.io/respx/
# - httpx testing guide: https://www.python-httpx.org/advanced/mocking-alternatives/
# - pytest fixtures, autouse: https://docs.pytest.org/en/stable/how-to/fixtures.html#autouse-fixtures-fixtures-you-don-t-have-to-request

import hashlib

import httpx
import respx

from app.infrastructure.search.extractor import (
    FETCH_TIMEOUT_SECONDS,
    _cache_key,
    extract_page_text,
)

_URL = "https://example.com/article"

_ARTICLE_HTML = """
<html><body><article><h1>Title</h1><p>This is the actual article body
content, long enough for trafilatura to treat it as the main content
block rather than boilerplate.</p></article></body></html>
"""

_EMPTY_HTML = "<html><body></body></html>"


@respx.mock
async def test_extract_page_text_returns_clean_text_on_success():
    # Ref: respx.get().mock(return_value=...) — https://lundberg.github.io/respx/api/#mock
    respx.get(_URL).mock(return_value=httpx.Response(200, html=_ARTICLE_HTML))

    text = await extract_page_text(_URL)

    assert text is not None
    assert "actual article body" in text
    # Boilerplate that isn't there shouldn't appear either — a loose
    # sanity check that trafilatura, not a naive strip, did the work.
    assert "Menu" not in text


@respx.mock
async def test_extract_page_text_returns_none_when_no_extractable_content():
    respx.get(_URL).mock(return_value=httpx.Response(200, html=_EMPTY_HTML))

    text = await extract_page_text(_URL)

    assert text is None


@respx.mock
async def test_extract_page_text_returns_none_on_http_error_status():
    respx.get(_URL).mock(return_value=httpx.Response(404))

    text = await extract_page_text(_URL)

    assert text is None


@respx.mock
async def test_extract_page_text_returns_none_on_timeout():
    # Ref: httpx exception classes used with respx's side_effect —
    # https://www.python-httpx.org/exceptions/
    respx.get(_URL).mock(side_effect=httpx.TimeoutException("timed out"))

    text = await extract_page_text(_URL)

    assert text is None


@respx.mock
async def test_extract_page_text_returns_none_on_connection_error():
    respx.get(_URL).mock(side_effect=httpx.ConnectError("refused"))

    text = await extract_page_text(_URL)

    assert text is None


@respx.mock
async def test_extract_page_text_caches_successful_result():
    # Ref: respx route.call_count — https://lundberg.github.io/respx/api/#routes
    route = respx.get(_URL).mock(return_value=httpx.Response(200, html=_ARTICLE_HTML))

    first = await extract_page_text(_URL)
    second = await extract_page_text(_URL)

    assert first == second
    # The whole point of the cache: the second call must not hit the network again.
    assert route.call_count == 1


@respx.mock
async def test_extract_page_text_caches_negative_result_too():
    # Pages with nothing extractable are cached as well (as _NO_CONTENT
    # internally), so a URL that keeps showing up in search results
    # doesn't get re-fetched every time just because it never had usable
    # content.
    route = respx.get(_URL).mock(return_value=httpx.Response(200, html=_EMPTY_HTML))

    first = await extract_page_text(_URL)
    second = await extract_page_text(_URL)

    assert first is None
    assert second is None
    assert route.call_count == 1


async def test_extract_page_text_uses_bounded_timeout():
    # Ref: httpx timeouts — https://www.python-httpx.org/advanced/timeouts/
    # Not a network test — just guards against someone accidentally
    # removing the explicit timeout in a future edit, which would let a
    # single slow page hang the whole search again.
    assert 0 < FETCH_TIMEOUT_SECONDS <= 30


def test_cache_key_is_stable_and_url_specific():
    # Ref: hashlib.sha256 — https://docs.python.org/3/library/hashlib.html
    assert _cache_key(_URL) == _cache_key(_URL)
    assert _cache_key(_URL) != _cache_key(_URL + "?utm_source=x")
    assert _cache_key(_URL) == hashlib.sha256(_URL.encode("utf-8")).hexdigest()
