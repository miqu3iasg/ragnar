# Testing

How tests are organized and what rules they follow. The goal is that any
new test added to this repo "looks like" the existing ones, fails for the
right reasons when it should, and never silently depends on real network
calls.

## Running tests

```bash
# Full suite (auto-collected, async-aware).
uv run pytest

# With line/branch coverage.
uv run pytest --cov=app

# A single file or test.
uv run pytest tests/domain/research/test_service.py
uv run pytest tests/domain/research/test_service.py::test_ask_returns_answer_built_from_completion
```

No coverage threshold is enforced today (see `pyproject.toml` — the
`[tool.coverage.*]` block configures reporting but doesn't gate). Use
`--cov` to see numbers; treat a sudden drop in coverage as a signal, not
a build failure.

## Layout

```
tests/
├── conftest.py                                  # shared autouse fixtures (cache reset)
├── api/
│   ├── test_exception_handlers.py               # domain-exception → HTTP status mapping
│   └── routes/
│       └── test_research.py                     # POST /ask end-to-end via TestClient
├── domain/
│   └── research/
│       ├── test_service.py                      # service-layer error translation, no RAG
│       └── test_service_rag.py                  # full RAG path with mocked boundaries
└── infrastructure/
    ├── embeddings/                              # sentence-transformers + cosine sim
    ├── llm/                                     # OpenRouter client + tenacity retry policy
    ├── rag/                                     # chunking, vector store, prompt builder
    └── search/                                  # Tavily + trafilatura
```

## Conventions

1. **No real network calls.** Every external dependency (OpenRouter,
   Tavily, httpx for page fetching, the local sentence-transformers
   model download) is mocked. The `respx` library is used for httpx
   mocks; `unittest.mock` (often `AsyncMock` + `patch`) for everything
   else.

2. **Patch where it's used, not where it's defined.** This is the
   "where to patch" rule from the `unittest.mock` docs. For example,
   `service.py` imports `get_completion_with_tools`; tests patch
   `app.domain.research.service.get_completion_with_tools`, not
   `app.infrastructure.llm.client.get_completion_with_tools`. Patching
   the definition site leaves the consumer's reference untouched and
   the real function still runs.

3. **Async tests use `asyncio_mode = "auto"`.** Set in
   `pyproject.toml`'s `[tool.pytest.ini_options]`. Every `async def
   test_*` is collected as an async test without needing
   `@pytest.mark.asyncio` or `pytestmark = pytest.mark.asyncio` at the
   top of the file.

4. **Process-lifetime state is reset between tests.** The embedding
   cache, the lazy-loaded embedding model, and the extraction cache
   are all module-level globals; without an autouse reset, tests leak
   state into each other. The reset fixtures live in
   `tests/conftest.py`. If you add a new module with module-level
   state, add a corresponding reset fixture there.

5. **Domain exception translation is its own test surface.** Tests in
   `tests/domain/research/test_service.py` prove that SDK exceptions
   (RateLimitError, APIConnectionError, …) become the right domain
   exception; tests in `tests/api/test_exception_handlers.py` prove
   those domain exceptions map to the right HTTP status. Both layers
   are tested independently so a failure points at one specific thing.

## Adding a new test

- Place it next to the module it covers. If you're testing
  `app/infrastructure/llm/client.py`, the test goes in
  `tests/infrastructure/llm/test_client.py`. Pytest only collects files
  matching `test_*.py` — a file named `text_extractor.py` will sit
  there forever, silently ignored.
- Match the existing style: heavy on `# Ref:` comments linking to the
  upstream docs for any non-obvious API, short docstrings on test
  helpers, no extraneous fixture ceremony beyond what `conftest.py`
  already provides.
- If your test needs a fresh module-level reset, extend `conftest.py`
  rather than duplicating the fixture in the test file.

## Why no coverage gate?

The codebase is small and the test suite has grown incrementally. A
hard threshold would either be set so low it's useless or so high that
every minor change costs a coverage PR. The signal is currently
visible — `uv run pytest --cov=app` shows the numbers — and the
decision to gate can be made once the suite stabilizes.

## Why `asyncio_mode = "auto"`?

It removes one class of footgun (forgetting `pytestmark` and getting
`RuntimeError: no running event loop` errors that look unrelated) at
the cost of one decorator-style `@pytest.mark.asyncio` left over in
older tests. Those leftover decorators are harmless no-ops under
`auto` mode and are left alone to minimize the diff in test files.
