# Refs (read these first, in this order, to understand the module as a whole):
# - sentence-transformers docs: https://www.sbert.net/
# - all-MiniLM-L6-v2 model card: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
# - cosine similarity for embeddings (background, not code-specific):
#   https://www.pinecone.io/learn/vector-similarity/
# - cachetools: https://cachetools.readthedocs.io/en/stable/
# - Python threading.Lock: https://docs.python.org/3/library/threading.html#lock-objects
# - asyncio.to_thread: https://docs.python.org/3/library/asyncio-task.html#asyncio.to_thread
#
# Embeddings are generated locally with sentence-transformers instead of
# through OpenRouter, for two reasons: (1) OPENROUTER_MODEL is a free-tier
# chat model, not an embeddings endpoint, so there's no embeddings API
# already wired up to reuse; (2) all-MiniLM-L6-v2 is small (~80MB) and
# fast enough on CPU that there's no real cost/latency case for calling
# an external API per chunk.

import asyncio
import hashlib
import logging
import threading

from cachetools import TTLCache
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

_MODEL_NAME = "all-MiniLM-L6-v2"

# Refs:
# - Double-checked locking pattern (why the None check happens both
#   outside and inside the lock): https://en.wikipedia.org/wiki/Double-checked_locking
# - threading.Lock vs asyncio.Lock — why this must be a real threading.Lock:
#   https://docs.python.org/3/library/asyncio-sync.html#asyncio.Lock
#   (asyncio.Lock only protects coroutines on the same event loop; the
#   model load below runs on a separate OS thread via asyncio.to_thread,
#   which asyncio.Lock does not guard against)
#
# The model used to be loaded at import time. That meant a slow or failed
# download (no network, HF rate limit, disk full) crashed the entire
# application on startup — before a single request had a chance to run,
# including requests that don't even touch retrieval. It's now loaded
# lazily, on first use. _model_lock guards specifically against two
# concurrent requests both finding _model is None and racing to load it
# on two different worker threads (see _embed_sync).
_model: SentenceTransformer | None = None
_model_lock = threading.Lock()


class EmbeddingModelUnavailableError(RuntimeError):
    """Raised when the local embedding model fails to load or run."""


def _get_model() -> SentenceTransformer:
    global _model
    if _model is not None:
        return _model

    with _model_lock:
        if _model is None:  # re-check: another thread may have just finished loading it
            try:
                logger.info(f"Loading embedding model '{_MODEL_NAME}'...")
                # Ref: SentenceTransformer constructor / model download & caching
                # behavior: https://www.sbert.net/docs/sentence_transformer/usage/usage.html
                _model = SentenceTransformer(_MODEL_NAME)
            except Exception as exc:
                # Ref (exception chaining, "raise ... from exc" / __cause__):
                # https://docs.python.org/3/tutorial/errors.html#exception-chaining
                logger.exception(f"Failed to load embedding model '{_MODEL_NAME}'.")
                raise EmbeddingModelUnavailableError(
                    f"Embedding model '{_MODEL_NAME}' could not be loaded."
                ) from exc
    return _model


# Refs:
# - cachetools.TTLCache: https://cachetools.readthedocs.io/en/stable/#cachetools.TTLCache
# - Why no lock is needed here (contrast with _model_lock above): all
#   access to this cache happens on the event loop thread, never inside
#   _embed_sync's worker thread — asyncio coroutines are cooperatively
#   scheduled, so a dict mutation with no `await` inside it can't be
#   interleaved with another coroutine's:
#   https://docs.python.org/3/library/asyncio-dev.html#concurrency-and-multithreading
# - dict.fromkeys() for order-preserving de-duplication:
#   https://docs.python.org/3/library/stdtypes.html#dict.fromkeys
#
# Repeated chunks (the same paragraph appearing on multiple pages, or the
# same source retrieved again for a similar question shortly after) are
# common enough that caching their embeddings avoids redundant CPU work.
# Keyed by a hash of the text, not the text itself, to keep cache keys a
# fixed size regardless of chunk length.
_CACHE_TTL_SECONDS = 3600
_embedding_cache: TTLCache = TTLCache(maxsize=4096, ttl=_CACHE_TTL_SECONDS)


def _cache_key(text: str) -> str:
    # Ref: hashlib docs — https://docs.python.org/3/library/hashlib.html
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _embed_sync(texts: list[str]) -> list[list[float]]:
    # Ref: SentenceTransformer.encode() parameters (convert_to_numpy,
    # normalize_embeddings): https://www.sbert.net/docs/package_reference/sentence_transformer/SentenceTransformer.html#sentence_transformers.SentenceTransformer.encode
    model = _get_model()
    embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return embeddings.tolist()


async def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Generate embeddings for a batch of texts, off the event loop.

    Ref (why normalize_embeddings=True lets cosine similarity collapse to
         a plain dot product downstream, in embeddings/ranking.py):
        https://www.pinecone.io/learn/vector-similarity/

    Results are cached per-text (see _embedding_cache above); only texts
    not already cached — including duplicates within the same batch — are
    sent to the model. Input order is preserved in the output regardless
    of which entries were cache hits.

    Raises EmbeddingModelUnavailableError if the local model can't be
    loaded or fails to run.
    """
    if not texts:
        return []

    keys = [_cache_key(text) for text in texts]
    unique_keys = list(dict.fromkeys(keys))  # de-dupe, keep first-seen order
    cached = {key: _embedding_cache.get(key) for key in unique_keys}

    missing_keys = [key for key in unique_keys if cached[key] is None]
    if missing_keys:
        key_to_text = {
            key: text for key, text in zip(keys, texts) if key in missing_keys
        }
        missing_texts = [key_to_text[key] for key in missing_keys]

        # Ref: asyncio.to_thread — https://docs.python.org/3/library/asyncio-task.html#asyncio.to_thread
        # Offloads the blocking (CPU-bound, and possibly first-call
        # model-loading) work so it doesn't stall the event loop.
        computed = await asyncio.to_thread(_embed_sync, missing_texts)
        for key, embedding in zip(missing_keys, computed):
            _embedding_cache[key] = embedding
            cached[key] = embedding

    return [cached[key] for key in keys]


async def embed_text(text: str) -> list[float]:
    """Convenience wrapper for embedding a single text (e.g. the user's question)."""
    embeddings = await embed_texts([text])
    return embeddings[0]
