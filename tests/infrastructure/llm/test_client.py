# Refs:
# - https://docs.pytest.org/en/stable/
# - https://docs.pytest.org/en/stable/how-to/usage.html
# - pytest-asyncio: https://pytest-asyncio.readthedocs.io/en/latest/
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from infrastructure.llm.client import get_completion


# Ref: https://pytest-asyncio.readthedocs.io/en/latest/how-to-guides/markers.html
@pytest.mark.asyncio
async def test_get_completion_returns_model_text():
    # Ref: https://docs.python.org/3/library/unittest.mock.html
    #
    # We are mocking the objects here to simulate what the OpenAI/OpenRouter API would actually return.
    # Since `completion.choices[0].message.content` is a chain of nested attributes, we use `MagicMock`,
    # which accepts any attribute without requiring us to define an actual class for it.
    fake_completion = MagicMock()
    fake_completion.choices[0].message.content = "Simulated response by AI"

    # Refs:
    # - where to patch: https://realpython.com/python-mock-library/#knowing-where-to-patch
    # - patch documentation: https://docs.python.org/3/library/unittest.mock-examples.html#mocking-classes
    # - AsyncMock: https://docs.python.org/3/library/unittest.mock.html#unittest.mock.AsyncMock
    #
    # patch() temporarily replaces the specified object, ONLY during
    # the "with" block. Note the path used: it is not "openai.AsyncOpenAI",
    # but rather where the client is USED within the infrastructure.llm.client module.
    with patch("infrastructure.llm.client.client") as mock_client:
        # We need to use AsyncMock here because a simple MagicMock would return itself,
        # instead of "awaitable" and the `await` inside get_completion would failed.
        mock_client.chat.completions.create = AsyncMock(return_value=fake_completion)

        result = await get_completion("What is the captal of Brazil?")

        assert result == "Simulated response by AI"

        mock_client.chat.completions.create.assert_called_once()

        _, kwargs = mock_client.chat.completions.create.call_args
        assert kwargs["messages"][0]["content"] == "What is the captal of Brazil?"
