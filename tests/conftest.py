# Refs:
# - pytest fixtures, autouse: https://docs.pytest.org/en/stable/how-to/fixtures.html#autouse-fixtures-fixtures-you-don-t-have-to-request
# - conftest.py discovery: https://docs.pytest.org/en/stable/reference/explanation/pythonpath.html#pythonpath-and-conftest-py-files
#
# Module-level autouse fixtures shared across the test suite.
#
# Several infrastructure modules keep process-lifetime state on module
# globals (lazy-loaded embedding model, embedding cache, extraction cache).
# Without an autouse reset, tests that mutate that state leak into each
# other: a fake model loaded for one test would still be there for the
# next, and a cached "no content" sentinel would short-circuit a later
# test that expected to actually fetch.
#
# Each fix here removes a duplicated fixture from an individual test file.
# Adding new ones follows the same pattern: import the module, snapshot /
# clear its globals around the test.

import pytest

from app.infrastructure.embeddings import client as embeddings_client
from app.infrastructure.search import extractor as extractor_module


@pytest.fixture(autouse=True)
def _reset_embedding_state():
    """Clear the lazy-loaded embedding model and its TTL cache per test."""
    embeddings_client._model = None
    embeddings_client._embedding_cache.clear()
    yield
    embeddings_client._model = None
    embeddings_client._embedding_cache.clear()


@pytest.fixture(autouse=True)
def _reset_extraction_cache():
    """Clear the extraction cache per test so cache-hit assertions are
    hermetic — a "first" call observed in one test must not leak into a
    "second" call observed in the next."""
    extractor_module._extraction_cache.clear()
    yield
    extractor_module._extraction_cache.clear()
