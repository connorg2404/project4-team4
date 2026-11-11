"""Build source-linked chunks without cutting sentences to meet a token cap.

Output is the team's nine-field schema. The report records omissions and short
chunks for review; it is not an assertion of retrieval quality or language purity.
"""

from collections import Counter
from statistics import median
import re

import pysbd

from app.embedding_config import (
    DOCUMENT_PREFIX, EMBEDDING_DIMENSIONS, EMBEDDING_MODEL, EMBEDDING_REVISION,
    MAX_CHUNK_TOKENS, MIN_FRAGMENT_TOKENS, QUERY_PREFIX, document_input,
)
from app.services.cleaning_service import clean_records, clean_text


def load_tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        EMBEDDING_MODEL, revision=EMBEDDING_REVISION,
        use_fast=True, trust_remote_code=False,
    )


def token_count(tokenizer, text: str) -> int:
    # Include the embedding task prefix, context header, and special tokens.
    return len(tokenizer.encode(document_input(text), add_special_tokens=True))


def _body_text(text: str) -> str:
    """Remove Markdown reference syntax without removing the readable labels."""
    text = re.sub(r"(?m)^\s*\[[^\]\n]+\]:\s*https?://\S+[^\n]*$", "", text)
    text = re.sub(r"\[([^\]\n]+)\]\[[^\]\n]*\]", r"\1", text)
    return clean_text(text)


def build_chunks(records, tokenizer, max_tokens=MAX_CHUNK_TOKENS):
    if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens < 50:
        raise ValueError("max_tokens must be an integer of at least 50.")
    cleaned, cleaning_report = clean_records(records)
    segmenter = pysbd.Segmenter(language="en", clean=False)
    chunks = []
    issues = []
    lengths = []
    short_chunks = []

    for record in cleaned:
        base = f"Title: {record['title']}\nAuthors: {', '.join(record['authors'])}"
        if record["first_publish_year"] is not None:
            base += f"\nFirst publication year (source): {record['first_publish_year']}"

        sections = [("description", _body_text(record["description"]), "Description")]
        if record["subjects"]:
            # Each tag remains a complete unit, even for very long subject lists.
            sections.append(("subjects", record["subjects"], "Subjects"))

        bio = record.get("author_bio")
        if bio:
            owner = record.get("author_bio_author")
            if isinstance(owner, str):
                owner = " ".join(clean_text(owner).split())
            if owner and owner in record["authors"]:
                sections.append(("author_bio", _body_text(bio), f"Biography of {owner}"))
            else:
                issues.append({
                    "work_id": record["work_id"], "section": "author_bio",
                    "reason": "Biography omitted: missing or unmatched author_bio_author.",
                })

        for section, body, label in sections:
            header = base + f"\n{label}:\n"
            if token_count(tokenizer, header) >= max_tokens:
                issues.append({"work_id": record["work_id"], "section": section,
                               "reason": "Context header leaves no room for section text."})
                continue
            units = []
            if section == "subjects":
                units = list(body)
            else:
                # Prefer intact paragraphs. Split an oversized paragraph into
                # sentences with abbreviation-aware segmentation.
                for paragraph in re.split(r"\n\s*\n", body):
                    paragraph = paragraph.strip()
                    if not paragraph:
                        continue
                    if token_count(tokenizer, header + paragraph) <= max_tokens:
                        units.append(paragraph)
                    else:
                        units.extend(s.strip() for s in segmenter.segment(paragraph) if s.strip())

            separator = "; " if section == "subjects" else "\n\n"
            groups = []
            current = []
            for unit in units:
                size = token_count(tokenizer, header + unit)
                if size > max_tokens:
                    # Do not silently truncate a sentence. Preserve the raw
                    # source and report the exact omitted unit for human review.
                    if current:
                        groups.append(current)
                        current = []
                    issues.append({"work_id": record["work_id"], "section": section,
                                   "reason": "Single unit exceeds token cap; not split mid-sentence.",
                                   "tokens_with_context": size, "omitted_text": unit})
                    continue
                candidate = separator.join(current + [unit])
                if current and token_count(tokenizer, header + candidate) > max_tokens:
                    groups.append(current)
                    current = [unit]
                else:
                    current.append(unit)
            if current:
                groups.append(current)

            # Balance a short final fragment using intact neighboring units.
            if len(groups) > 1:
                previous, final = groups[-2], groups[-1]
                combined = separator.join(previous + final)
                if token_count(tokenizer, header + combined) <= max_tokens:
                    groups[-2:] = [previous + final]
                else:
                    while len(previous) > 1 and len(tokenizer.encode(separator.join(final), add_special_tokens=False)) < MIN_FRAGMENT_TOKENS:
                        candidate = [previous[-1]] + final
                        if token_count(tokenizer, header + separator.join(candidate)) > max_tokens:
                            break
                        final.insert(0, previous.pop())

            for index, group in enumerate(groups):
                body_part = separator.join(group)
                text = header + body_part
                size = token_count(tokenizer, text)
                chunk = {
                    "chunk_id": f"{record['work_id']}::{section}::{index}",
                    "text": text,
                    "work_id": record["work_id"],
                    "title": record["title"],
                    "authors": record["authors"],
                    "subjects": record["subjects"],
                    "section": section,
                    "first_publish_year": record["first_publish_year"],
                    "source_url": record["source_url"],
                }
                chunks.append(chunk)
                lengths.append(size)
                if len(tokenizer.encode(body_part, add_special_tokens=False)) < MIN_FRAGMENT_TOKENS:
                    short_chunks.append(chunk["chunk_id"])

    report = {
        "stage": "chunking_only_no_embeddings",
        "embedding_model": EMBEDDING_MODEL,
        "embedding_revision": EMBEDDING_REVISION,
        "embedding_dimensions": EMBEDDING_DIMENSIONS,
        "document_prefix_added_by_indexer": DOCUMENT_PREFIX,
        "query_prefix_added_by_retriever": QUERY_PREFIX,
        "max_tokens_including_prefix_context_and_special_tokens": max_tokens,
        "cleaning": cleaning_report,
        "chunk_count": len(chunks),
        "works_with_chunks": len({c["work_id"] for c in chunks}),
        "section_counts": dict(Counter(c["section"] for c in chunks)),
        "token_length": {"min": min(lengths, default=0), "median": median(lengths) if lengths else 0,
                         "max": max(lengths, default=0)},
        "token_histogram": {"under_200": sum(n < 200 for n in lengths),
                            "200_to_400": sum(200 <= n <= 400 for n in lengths),
                            "over_400": sum(n > 400 for n in lengths)},
        "issues": issues,
        "short_body_chunk_ids": short_chunks,
        "limitations": [
            "No embedding vectors, vector database, or retrieval evaluation built here.",
            "English sentence segmentation; source languages are preserved, not detected or filtered.",
            "Short sections are retained and reported; cross-section merging needs human review.",
            "Overlong indivisible units are omitted and included verbatim in issues, not truncated.",
            "No publication-year verification or unsupported bestseller/stock claims added.",
            "Chunk ordinals are stable for identical inputs/config; changed data requires index reconciliation.",
        ],
    }
    return chunks, report
