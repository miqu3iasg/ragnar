# Refs:
# - Martin Fowler, "Domain Model" (translating infra exceptions to domain
#   exceptions at a service boundary): https://martinfowler.com/eaaCatalog/domainModel.html
# - Python tutorial, "Exceptions" (defining custom exception hierarchies):
#   https://docs.python.org/3/tutorial/errors.html#user-defined-exceptions
# - Python tutorial, "Exception Chaining" (the "raise ... from exc" pattern
#   used throughout service.py when raising these):
#   https://docs.python.org/3/tutorial/errors.html#exception-chaining
#
# Domain-level exceptions for the research flow.
#
# Rationale: the API layer (FastAPI routes) should never know about SDK-specific
# exception types like openai.RateLimitError. If we ever swap providers or SDKs,
# only infrastructure/llm/client.py and this translation boundary in service.py
# should need to change, that is, routes stay untouched.
class ResearchError(Exception):
    """Base exception for the research domain. Catch this in the API layer
    as a fallback if a more specific exception type is ever missed."""


class EmptyQuestionError(ResearchError):
    """Raised when the the user submitted question is empty or filled just with white spaces."""


class LLMRateLimitError(ResearchError):
    """Raised when the LLm provider resturns a rate limit error and all
    retry attempts (we handled that by tenacity in infrastructure/llm/client.py)
    have been exhausted."""

    def __init__(
        self, message: str, retry_after: float | None = None
    ) -> None:
        super().__init__(message)
        # Best-effort copy of the provider-suggested wait time in seconds.
        # Used by api/exception_handlers.py's llm_rate_limit_handler to
        # surface a Retry-After HTTP header on the 429 response. None when
        # the provider didn't send one — handler then omits the header.
        self.retry_after = retry_after


class LLMUnavailableError(ResearchError):
    """Raised when the LLM provider is unreachable or returns a server side
    error (connection failure, timeout, 5xx) after retries are exhausted."""


class LLMResponseError(ResearchError):
    """Raised when the LLM call technically succeeds (no exception) but the
    response content is missing, empty, or otherwise unusable.
    This is distinct from LLMUnavailableError; the provider answered, but
    the answer itself is not valid. The common causes are: finish_reason=content_filter,
    finish_reason=length with an empty completion, or a model returning an
    empty string for other reasons.
    Ref: https://platform.openai.com/docs/api-reference/chat/object
    (see `finish_reason` field)
    """


class RetrievalUnavailableError(ResearchError):
    """Raised when the retrieval pipeline (web search or the local
    embedding model) fails outright, as opposed to a single source
    failing to fetch/extract, which is skipped rather than raised (see
    domain/research/service.py's _run_search_tool).

    Covers: the search provider (Tavily) being unreachable or erroring —
    ref: https://docs.tavily.com/documentation/api-reference/endpoint/search
    — and the local embedding model failing to load or run
    (EmbeddingModelUnavailableError from infrastructure/embeddings/
    client.py — ref: https://www.sbert.net/). Mirrors LLMUnavailableError's
    semantics one layer over: "a provider this request depends on is
    down", just for the retrieval side of the pipeline instead of the LLM
    call itself.
    """
