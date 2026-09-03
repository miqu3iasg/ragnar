# Refs:
# - tiktoken: https://github.com/openai/tiktoken
# - Chunking strategies for RAG: https://www.pinecone.io/learn/chunking-strategies/
#
# Chunking is done by token count, not character count, because both the
# embedding model and the LLM operate on tokens — a character-length limit
# doesn't translate reliably across languages or punctuation-heavy text.
# Overlap between consecutive chunks preserves context that would
# otherwise be cut at a chunk boundary (e.g. a sentence split in half).

import tiktoken

# cl100k_base isn't tied to any specific OpenRouter model, but it's a
# stable, widely available encoding that's good enough for sizing chunks
# consistently, we're not tokenizing for the LLM call itself here, only
# deciding where to cut the text.
_ENCODING = tiktoken.get_encoding("cl100k_base")


def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> list[str]:
    """
    Split text into overlapping chunks of roughly `chunk_size` tokens.

    chunk_size=300: small enough that several chunks fit in the final RAG
    prompt alongside the question, large enough to preserve coherent
    context (a paragraph or two) per chunk.
    overlap=50 (~17% of chunk_size): prevents losing a sentence that
    happens to fall right on a chunk boundary.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size.")

    tokens = _ENCODING.encode(text)
    if not tokens:
        return []

    chunks = []
    start = 0
    step = chunk_size - overlap

    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunks.append(_ENCODING.decode(tokens[start:end]))

        if end == len(tokens):
            break
        start += step

    return chunks
