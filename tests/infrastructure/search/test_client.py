from unittest.mock import AsyncMock, patch

import pytest

from app.infrastructure.search.client import search_web


@pytest.mark.asyncio
async def test_search_web_returns_parsed_results():
    fake_response = {
        "results": [
            {"url": "https://example.com/a", "title": "A", "content": "snippet a"},
            {"url": "https://example.com/b", "title": "B", "content": "snippet b"},
        ]
    }

    with patch(
        "app.infrastructure.search.client.client.search",
        new=AsyncMock(return_value=fake_response),
    ):
        results = await search_web("test query")

    assert len(results) == 2
    assert str(results[0].url) == "https://example.com/a"
    assert results[0].title == "A"
    assert results[0].snippet == "snippet a"


@pytest.mark.asyncio
async def test_search_web_skips_malformed_results():
    fake_response = {
        "results": [{"url": "not-a-valid-url", "title": "Bad", "content": "x"}]
    }

    with patch(
        "app.infrastructure.search.client.client.search",
        new=AsyncMock(return_value=fake_response),
    ):
        results = await search_web("test query")

    assert results == []
