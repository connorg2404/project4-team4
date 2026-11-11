"""Shared Nomic contract. Chunking uses the tokenizer, not model weights."""

EMBEDDING_MODEL = "nomic-ai/nomic-embed-text-v1.5"
EMBEDDING_REVISION = "e9b6763023c676ca8431644204f50c2b100d9aab"
EMBEDDING_DIMENSIONS = 768
DOCUMENT_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "
MAX_CHUNK_TOKENS = 400
MIN_FRAGMENT_TOKENS = 50


def document_input(text: str) -> str:
    """Call only on unprefixed chunk text; the indexer owns this prefix."""
    return DOCUMENT_PREFIX + text


def query_input(text: str) -> str:
    return QUERY_PREFIX + text
