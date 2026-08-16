# refs:
# https://docs.pytest.org/en/stable/
# https://docs.pytest.org/en/stable/how-to/usage.html
from unittest.mock import patch, MagicMock

from infrastructure.llm.client import get_completion


def test_get_completion_returns_model_text():
    # ref:
    # https://docs.python.org/3/library/unittest.mock.html
    # We are mocking the objects here to simulate what the OpenAI/OpenRouter API would actually return.
    # Since `completion.choices[0].message.content` is a chain of nested attributes, we use `MagicMock`,
    # which accepts any attribute without requiring us to define an actual class for it.
    fake_completion = MagicMock()
    fake_completion.choices[0].message.content = "Simulated response by AI"

    # refs:
    # where to patch: https://realpython.com/python-mock-library/#knowing-where-to-patch
    # patch documentation: https://docs.python.org/3/library/unittest.mock-examples.html#mocking-classes
    # available assertions: https://docs.python.org/3/library/unittest.mock.html#unittest.mock.Mock
    #
    # patch() temporarily replaces the specified object, ONLY during
    # the "with" block. Note the path used: it is not "openai.OpenAI",
    # but rather where the client is USED within the infrastructure.llm.client module.
    # This is the "where to patch" aspect that the article above mention.
    with patch("infrastructure.llm.client.client") as mock_client:
        mock_client.chat.completions.create.return_value = fake_completion

        result = get_completion("What is the captal of Brazil?")

        assert result == "Simulated response by AI"

        mock_client.chat.completions.create.assert_called_once()
        _, kwargs = mock_client.chat.completions.create.call_args
        assert kwargs["messages"][0]["content"] == "What is the captal of Brazil?"
