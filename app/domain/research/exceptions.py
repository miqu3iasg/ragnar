# Domain-level exceptions for the research flow.
#
# Rationale: the API layer (FastAPI routes) should never know about SDK-specific
# exception types like openai.RateLimitError. If we ever swap providers or SDKs,
# only infrastructure/llm/client.py and this translation boundary in service.py
# should need to change, that is, routes stay untouched.
#
# Pattern ref (translating infra exceptions to domain exceptions at a
# service boundary): https://martinfowler.com/eaaCatalog/domainModel.html


class ResearchError(Exception):
    """Base exception for the research domain. Catch this in the API layer
    as a fallback if a more specific exception type is ever missed."""


class EmptyQuestionError(ResearchError):
    """Raised when the the user submitted question is empty or filled just with white spaces."""


class LLMRateLimitError(ResearchError):
    """Raised when the LLm provider resturns a rate limit error and all
    retry attempts (we handled that by tenacity in infrastructure/llm/client.py)
    have been exhausted."""


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
