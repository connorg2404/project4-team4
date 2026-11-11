"""Run from the project root: python -m scripts.index_chunks --input PATH.

Embeds chunk_records output with the pinned Nomic model and upserts it into Chroma.
"""

import argparse
import hashlib
from pathlib import Path

from app.services import vectordb_service
from scripts.clean_records import read_jsonl

REQUIRED_FIELDS = {
    "chunk_id": str, "text": str, "work_id": str, "title": str, "authors": list,
    "subjects": list, "section": str, "source_url": str,
}


def validate(chunks: list[dict]) -> None:
    """Fail before loading the model if any chunk doesn't match the chunker's schema."""
    seen = set()
    for line_number, chunk in enumerate(chunks, 1):
        where = f"Chunk {line_number} ({chunk.get('chunk_id', 'no chunk_id')})"
        for field, kind in REQUIRED_FIELDS.items():
            if not isinstance(chunk.get(field), kind):
                raise ValueError(f"{where}: '{field}' must be a {kind.__name__}.")
        year = chunk.get("first_publish_year")
        if year is not None and (not isinstance(year, int) or isinstance(year, bool)):
            raise ValueError(f"{where}: 'first_publish_year' must be an integer or null.")
        if not chunk["text"].strip():
            raise ValueError(f"{where}: empty text.")
        if chunk["chunk_id"] in seen:
            raise ValueError(f"{where}: duplicate chunk_id.")
        seen.add(chunk["chunk_id"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed chunks and upsert them into the Chroma vector DB.")
    parser.add_argument("--input", type=Path, required=True, help="JSONL from scripts.chunk_records")
    parser.add_argument("--collection", default=vectordb_service.DEFAULT_COLLECTION)
    parser.add_argument("--reset", action="store_true",
                        help="Delete the collection first, so it contains exactly this file's chunks.")
    parser.add_argument("--dry-run", action="store_true", help="Validate the input without embedding anything.")
    args = parser.parse_args()

    try:
        chunks = list(read_jsonl(args.input))
        if not chunks:
            parser.error("Input contains no chunks.")
        validate(chunks)
    except (OSError, ValueError) as error:
        parser.error(str(error))

    print(f"Input: {args.input} | sha256 {hashlib.sha256(args.input.read_bytes()).hexdigest()}")
    print(f"Chunks: {len(chunks)} from {len({c['work_id'] for c in chunks})} books")
    if args.dry_run:
        print("Dry run: input is valid; nothing was embedded.")
        return

    try:
        if args.reset:
            vectordb_service.reset_collection(args.collection)
            print(f"Reset collection '{args.collection}'.")
        indexed = vectordb_service.ingest_chunks(chunks, args.collection)
        collection = vectordb_service.create_or_get_collection(args.collection)
        stale = set(collection.get(include=[])["ids"]) - {c["chunk_id"] for c in chunks}
    except (OSError, ValueError) as error:
        parser.error(str(error))

    print(f"Upserted: {indexed} | Collection '{args.collection}' now holds {collection.count()} chunks")
    if stale:
        print(f"Warning: {len(stale)} chunks in the collection are not in this file (from an earlier run). "
              "Re-run with --reset to remove them.")
    print(f"Vector DB: {Path(vectordb_service.PERSIST_DIRECTORY).resolve()}")


if __name__ == "__main__":
    main()
