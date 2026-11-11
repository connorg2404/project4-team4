"""Chroma vector store for book chunks produced by scripts.chunk_records.

Embeddings come from the pinned Nomic model in app.embedding_config, so the
index matches the tokenizer the chunker measured with. This module owns the
Nomic task prefixes: "search_document: " on ingest, "search_query: " on search.
"""

from functools import lru_cache
from typing import Any, Iterable

import chromadb

from app.embedding_config import (
    DOCUMENT_PREFIX, EMBEDDING_DIMENSIONS, EMBEDDING_MODEL, EMBEDDING_REVISION,
    QUERY_PREFIX, document_input, query_input,
)

PERSIST_DIRECTORY = "app/chroma_db"  # Relative to the project root (run from there)
DEFAULT_COLLECTION = "books"
EMBED_BATCH_SIZE = 32

# Stored on the collection so an index built with one model is never queried with another.
_INDEX_CONTRACT = {
    "embedding_model": EMBEDDING_MODEL,
    "embedding_revision": EMBEDDING_REVISION,
    "embedding_dimensions": EMBEDDING_DIMENSIONS,
}

# Chunk fields copied into Chroma metadata. Chroma stores scalars and non-empty string lists.
_LIST_FIELDS = ("authors", "subjects")
_SCALAR_FIELDS = ("work_id", "title", "section", "first_publish_year", "source_url")


@lru_cache(maxsize=1)
def _model():
    """Load the embedding model once, on first use (not at import time)."""
    from sentence_transformers import SentenceTransformer

    # Nomic's architecture is custom code on the Hub, so trust_remote_code is required.
    # The pinned revision keeps the weights and config fixed.
    # device="cpu": PyTorch's Apple GPU (MPS) backend crashes the process on some Macs.
    return SentenceTransformer(
        EMBEDDING_MODEL, revision=EMBEDDING_REVISION, trust_remote_code=True, device="cpu",
    )


def _embed(texts: list[str]) -> list[list[float]]:
    vectors = _model().encode(
        texts, batch_size=EMBED_BATCH_SIZE, normalize_embeddings=True, show_progress_bar=len(texts) > 100,
    )
    if vectors.shape[1] != EMBEDDING_DIMENSIONS:
        raise ValueError(f"Model returned {vectors.shape[1]} dimensions; expected {EMBEDDING_DIMENSIONS}.")
    return vectors.tolist()


@lru_cache(maxsize=1)
def _client():
    return chromadb.PersistentClient(path=PERSIST_DIRECTORY)


@lru_cache(maxsize=None)
def create_or_get_collection(collection_name: str = DEFAULT_COLLECTION):
    """Open a collection (like a SQL table), creating it on first use."""
    collection = _client().get_or_create_collection(
        name=collection_name,
        embedding_function=None,  # We always pass vectors ourselves
        configuration={"hnsw": {"space": "cosine"}},
        metadata=_INDEX_CONTRACT,
    )
    stored = {key: (collection.metadata or {}).get(key) for key in _INDEX_CONTRACT}
    if stored != _INDEX_CONTRACT:
        raise ValueError(
            f"Collection '{collection_name}' was built with {stored}, but this code uses {_INDEX_CONTRACT}. "
            "Rebuild it with reset_collection() and re-ingest."
        )
    return collection


def reset_collection(collection_name: str = DEFAULT_COLLECTION) -> None:
    """Delete a collection so it can be rebuilt from scratch."""
    try:
        _client().delete_collection(collection_name)
    except Exception:  # Chroma raises different errors by version when it is missing
        pass
    create_or_get_collection.cache_clear()


def _metadata(chunk: dict[str, Any]) -> dict[str, Any]:
    meta = {}
    for field in _SCALAR_FIELDS:
        value = chunk.get(field)
        if value is not None:  # Chroma rejects None; a missing year stays absent, not guessed
            meta[field] = value
    for field in _LIST_FIELDS:
        values = chunk.get(field)
        if values:  # Kept as real lists ("Spyri, Johanna" must stay one author); Chroma rejects []
            meta[field] = list(values)
    return meta


def ingest_chunks(chunks: Iterable[dict[str, Any]], collection_name: str = DEFAULT_COLLECTION) -> int:
    """Embed and upsert chunk_records output. Upserting by chunk_id makes re-runs safe."""
    chunks = list(chunks)
    ids = [chunk["chunk_id"] for chunk in chunks]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate chunk_id values in input.")
    for chunk in chunks:
        if chunk["text"].startswith((DOCUMENT_PREFIX, QUERY_PREFIX)):
            raise ValueError(f"{chunk['chunk_id']} is already prefixed; pass unprefixed chunk text.")

    collection = create_or_get_collection(collection_name)
    batch_size = _client().get_max_batch_size()
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start:start + batch_size]
        collection.upsert(
            ids=[chunk["chunk_id"] for chunk in batch],
            embeddings=_embed([document_input(chunk["text"]) for chunk in batch]),
            documents=[chunk["text"] for chunk in batch],  # Stored without the prefix
            metadatas=[_metadata(chunk) for chunk in batch],
        )
    return len(chunks)


def search_collection(
    query: str,
    k: int = 10,
    collection_name: str = DEFAULT_COLLECTION,
    where: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Return the k nearest chunks, closest first.

    `where` is an optional Chroma metadata filter, e.g. {"section": "description"}
    or {"work_id": "OL21745884W"}. `distance` is cosine distance (lower is closer).
    """
    if not query.strip():
        raise ValueError("Query must not be empty.")
    collection = create_or_get_collection(collection_name)
    results = collection.query(
        query_embeddings=_embed([query_input(query)]),
        n_results=k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )
    return [
        {"chunk_id": chunk_id, "text": text, "metadata": metadata, "distance": distance}
        for chunk_id, text, metadata, distance in zip(
            results["ids"][0], results["documents"][0], results["metadatas"][0], results["distances"][0],
        )
    ]


def collection_count(collection_name: str = DEFAULT_COLLECTION) -> int:
    return create_or_get_collection(collection_name).count()


def warm_up() -> None:
    """Load the embedding model now, so the first request doesn't pay for it."""
    _embed([query_input("warm up")])
