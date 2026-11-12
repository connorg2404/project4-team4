"""Add one user-supplied book to the vector index.

Runs the same cleaning and chunking as the Open Library books, then embeds and
upserts the chunks into the same Chroma collection, so /ask can retrieve it.
"""

import uuid
from functools import lru_cache
from typing import Any, Callable


@lru_cache(maxsize=1)
def _tokenizer():
    """Load the pinned Nomic tokenizer once, on the first request (it is downloaded then)."""
    from app.services.chunking_service import load_tokenizer

    return load_tokenizer()


def new_work_id() -> str:
    return "USER-" + uuid.uuid4().hex[:12]


def add_book(book: dict[str, Any], tokenizer=None, ingest: Callable | None = None) -> dict[str, Any]:
    """Chunk and index `book` (title, authors, description, and optional subjects,
    first_publish_year, author_bio, author_bio_author). Raises ValueError if the
    book produces no chunks. `tokenizer` and `ingest` are for tests."""
    from app.services.chunking_service import build_chunks

    if ingest is None:
        from app.services.vectordb_service import ingest_chunks as ingest

    record = {
        "work_id": new_work_id(),
        "title": book["title"],
        "authors": book["authors"],
        "author_bio": book.get("author_bio"),
        "author_bio_author": book.get("author_bio_author"),
        "description": book["description"],
        "subjects": book.get("subjects") or [],
        "first_publish_year": book.get("first_publish_year"),
        "source_url": None,  # no Open Library page for a user-added book
        "popularity": 0,
        "cover_id": None,
    }
    chunks, report = build_chunks([record], tokenizer or _tokenizer())
    if not chunks:
        reasons = [issue["reason"] for issue in report.get("issues", [])]
        raise ValueError("The book produced no chunks to index. " + " ".join(reasons))
    indexed = ingest(chunks)
    return {
        "work_id": record["work_id"],
        "title": chunks[0]["title"],
        "chunks_added": indexed,
        "sections": sorted({chunk["section"] for chunk in chunks}),
    }
