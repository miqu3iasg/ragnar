# Refs:
# - unittest.mock, "Where to patch" (the single most important concept for getting these patches to actually take effect): https://docs.python.org/3/library/unittest.mock.html#where-to-patch
# - unittest.mock.MagicMock / side_effect: https://docs.python.org/3/library/unittest.mock.html#unittest.mock.Mock.side_effect
# - pytest-asyncio markers: https://pytest-asyncio.readthedocs.io/en/stable/reference/markers/index.html
# - pytest fixtures, autouse: https://docs.pytest.org/en/stable/how-to/fixtures.html#autouse-fixtures-fixtures-you-don-t-have-to-request
# - numpy.array (used to fake SentenceTransformer.encode's return shape): https://numpy.org/doc/stable/reference/generated/numpy.array.html

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.infrastructure.embeddings.client import (
    EmbeddingModelUnavailableError,
    embed_text,
    embed_texts,
)


def _fake_model(vector_for=lambda text: [float(len(text)), 0.0, 0.0]) -> MagicMock:
    # Ref: SentenceTransformer.encode() real signature/return shape (we're faking it here, not calling the real thing):
    # https://www.sbert.net/docs/package_reference/sentence_transformer/SentenceTransformer.html#sentence_transformers.SentenceTransformer.encode
    model = MagicMock()

    def encode(texts, convert_to_numpy=True, normalize_embeddings=True):
        return np.array([vector_for(text) for text in texts])

    model.encode.side_effect = encode
    return model


async def test_embed_texts_returns_empty_list_for_empty_input():
    with patch("app.infrastructure.embeddings.client.SentenceTransformer") as mock_cls:
        result = await embed_texts([])

    assert result == []
    mock_cls.assert_not_called()  # loading the model at all would be wasted work


async def test_embed_texts_loads_model_lazily_on_first_use():
    with patch(
        "app.infrastructure.embeddings.client.SentenceTransformer",
        return_value=_fake_model(),
    ) as mock_cls:
        mock_cls.assert_not_called()  # not loaded at import time
        await embed_texts(["hello"])
        mock_cls.assert_called_once_with("all-MiniLM-L6-v2")


async def test_embed_texts_preserves_input_order():
    with patch(
        "app.infrastructure.embeddings.client.SentenceTransformer",
        return_value=_fake_model(),
    ):
        embeddings = await embed_texts(["a", "bb", "ccc"])

    assert [vec[0] for vec in embeddings] == [1.0, 2.0, 3.0]


async def test_embed_texts_only_computes_uncached_entries():
    fake_model = _fake_model()
    with patch(
        "app.infrastructure.embeddings.client.SentenceTransformer",
        return_value=fake_model,
    ):
        await embed_texts(["repeat me"])
        # Ref: Mock.reset_mock() :: https://docs.python.org/3/library/unittest.mock.html#unittest.mock.Mock.reset_mock
        fake_model.encode.reset_mock()

        await embed_texts(["repeat me", "new text"])

    # Only "new text" should have gone to the model; "repeat me" came from cache.
    (called_texts,), _kwargs = fake_model.encode.call_args
    assert called_texts == ["new text"]


async def test_embed_texts_deduplicates_repeats_within_the_same_batch():
    fake_model = _fake_model()
    with patch(
        "app.infrastructure.embeddings.client.SentenceTransformer",
        return_value=fake_model,
    ):
        embeddings = await embed_texts(["same", "same", "same"])

    (called_texts,), _kwargs = fake_model.encode.call_args
    assert called_texts == ["same"]  # computed once, not three times
    assert (
        embeddings[0] == embeddings[1] == embeddings[2]
    )  # returned for every occurrence


async def test_embed_text_returns_single_vector():
    with patch(
        "app.infrastructure.embeddings.client.SentenceTransformer",
        return_value=_fake_model(),
    ):
        embedding = await embed_text("solo")

    assert embedding == [4.0, 0.0, 0.0]  # len("solo") == 4


async def test_embed_texts_raises_domain_exception_when_model_fails_to_load():
    with patch(
        "app.infrastructure.embeddings.client.SentenceTransformer",
        side_effect=OSError("could not download model weights"),
    ):
        with pytest.raises(EmbeddingModelUnavailableError):
            await embed_texts(["hello"])


async def test_embed_texts_chains_original_exception_as_cause():
    # Ref: exception chaining / __cause__ — https://docs.python.org/3/tutorial/errors.html#exception-chaining
    original_error = OSError("could not download model weights")
    with patch(
        "app.infrastructure.embeddings.client.SentenceTransformer",
        side_effect=original_error,
    ):
        with pytest.raises(EmbeddingModelUnavailableError) as exc_info:
            await embed_texts(["hello"])

    assert exc_info.value.__cause__ is original_error


async def test_embed_texts_retries_after_a_failed_load():
    # A failed load isn't cached as a value (there's nothing to cache), so
    # a later call gets a fresh attempt instead of replaying a stale
    # failure, useful if e.g. a transient network blip caused the first
    # failure.
    call_count = 0

    def flaky_constructor(_name):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise OSError("transient failure")
        return _fake_model()

    with patch(
        "app.infrastructure.embeddings.client.SentenceTransformer",
        side_effect=flaky_constructor,
    ):
        with pytest.raises(EmbeddingModelUnavailableError):
            await embed_texts(["hello"])

        embeddings = await embed_texts(["hello"])

    assert embeddings == [[5.0, 0.0, 0.0]]
    assert call_count == 2


async def test_embed_texts_wraps_encode_failure_as_domain_exception():
    # The SentenceTransformer constructor can succeed but a later
    # .encode() call can still raise (OOM, dimension mismatch on a
    # poisoned cache, a corrupted weight file). That raw exception must
    # be wrapped in EmbeddingModelUnavailableError so the service layer's
    # catch is uniform with load failures, otherwise it would leak as an
    # unhandled exception.
    failing_model = MagicMock()
    failing_model.encode.side_effect = RuntimeError("encode blew up")

    with patch(
        "app.infrastructure.embeddings.client.SentenceTransformer",
        return_value=failing_model,
    ):
        with pytest.raises(EmbeddingModelUnavailableError, match="failed while generating"):
            await embed_texts(["hello"])


async def test_embed_texts_chains_original_encode_exception_as_cause():
    # Same exception-chaining invariant as test_embed_texts_chains_original_
    # exception_as_cause, but for the encode path: when a model.encode
    # call raises, the wrapper preserves __cause__ so the underlying
    # exception is recoverable from logs / Sentry.
    original_error = RuntimeError("encode blew up")
    failing_model = MagicMock()
    failing_model.encode.side_effect = original_error

    with patch(
        "app.infrastructure.embeddings.client.SentenceTransformer",
        return_value=failing_model,
    ):
        with pytest.raises(EmbeddingModelUnavailableError) as exc_info:
            await embed_texts(["hello"])

    assert exc_info.value.__cause__ is original_error


async def test_embed_text_does_not_call_model_for_already_cached_text():
    # Cache-hit short-circuit: the second call for the same text must
    # NOT reach the model. This pins the contract that the TTL cache is
    # actually consulted before any encode work is scheduled.
    fake_model = _fake_model()

    with patch(
        "app.infrastructure.embeddings.client.SentenceTransformer",
        return_value=fake_model,
    ):
        first = await embed_text("repeat me")
        fake_model.encode.reset_mock()
        second = await embed_text("repeat me")

    assert first == second
    fake_model.encode.assert_not_called()
