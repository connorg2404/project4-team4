"""Conservative cleanup of Person 1's Open Library records.

This stage preserves raw metadata and missing years. It does not generate
chunks, detect language, or invent missing facts.
"""

from collections import Counter
from html import unescape
from html.parser import HTMLParser
import re
from typing import Any, Iterable


class InvalidRecord(ValueError):
    """A record cannot safely be passed to the next stage."""


class _TextExtractor(HTMLParser):
    BLOCK_TAGS = {
        "p", "div", "br", "li", "ul", "ol", "section", "blockquote",
        "h1", "h2", "h3", "h4", "h5", "h6",
    }

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.hidden_tag: str | None = None

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if self.hidden_tag:
            return
        if tag in {"script", "style"}:
            self.hidden_tag = tag
        elif tag in self.BLOCK_TAGS:
            self.parts.append("\n\n")

    def handle_endtag(self, tag: str) -> None:
        if self.hidden_tag:
            if tag == self.hidden_tag:
                self.hidden_tag = None
            return
        if tag in self.BLOCK_TAGS:
            self.parts.append("\n\n")

    def handle_data(self, data: str) -> None:
        if not self.hidden_tag:
            self.parts.append(data)


def clean_text(value: str | None) -> str:
    """Keep visible text while removing common HTML/Markdown formatting.

    Paragraph breaks and useful link labels are retained. Bare URLs and
    uncertain boilerplate are deliberately preserved for later review.
    """
    if value is None:
        return ""
    if not isinstance(value, str):
        raise InvalidRecord("Text fields must be strings or null.")

    parser = _TextExtractor()
    parser.feed(value)
    parser.close()
    text = unescape("".join(parser.parts))
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

    # Remove standalone decorative rules before Markdown cleanup.
    # Keep a paragraph boundary and preserve hyphens inside actual text.
    text = re.sub(
        r"(?m)^[^\S\n]*(?:-(?:[^\S\n]*-){2,}|"
        r"\*(?:[^\S\n]*\*){2,}|_(?:[^\S\n]*_){2,})[^\S\n]*$",
        "\n",
        text,
    )

    # Images and links retain their readable labels.
    text = re.sub(r"!?\[([^\]]+)\]\([^\n)]*\)", r"\1", text)
    text = re.sub(r"(?m)^\s{0,3}#{1,6}\s+", "", text)
    for pattern in (r"\*\*(.+?)\*\*", r"__(.+?)__", r"(?<!\w)\*([^*\n]+)\*(?!\w)", r"(?<!\w)_([^_\n]+)_(?!\w)", r"`([^`\n]+)`"):
        text = re.sub(pattern, r"\1", text)
    text = re.sub(r"(?m)^\s*[-+*]\s+", "", text)
    text = re.sub(r"[^\S\n]+", " ", text)
    text = "\n".join(line.strip() for line in text.splitlines())
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _string_list(value: Any, field: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise InvalidRecord(f"{field} must be a list of strings.")
    return value


def normalize_subjects(subjects: list[str]) -> list[str]:
    """Deduplicate tags and remove NYT tracking tags, preserving meaning."""
    result: list[str] = []
    seen: set[str] = set()
    for subject in _string_list(subjects, "subjects"):
        cleaned = " ".join(clean_text(subject).split()).casefold()
        if not cleaned or cleaned.startswith("nyt:") or cleaned in seen:
            continue
        result.append(cleaned)
        seen.add(cleaned)
    return result


def clean_record(record: dict[str, Any]) -> dict[str, Any]:
    """Return a new record; never alter the original or guess missing data."""
    if not isinstance(record, dict):
        raise InvalidRecord("Each record must be a JSON object.")

    work_id = record.get("work_id")
    if not isinstance(work_id, str) or not re.fullmatch(r"OL\d+W|USER-[0-9a-f]{12}", work_id.strip()):
        raise InvalidRecord("work_id must be an Open Library work ID or a USER-<12 hex digits> ID.")
    work_id = work_id.strip()
    title = " ".join(clean_text(record.get("title")).split())
    description = clean_text(record.get("description"))
    if not title:
        raise InvalidRecord("Title is missing after cleaning.")
    if not description:
        raise InvalidRecord("Description is missing after cleaning.")

    authors: list[str] = []
    for name in _string_list(record.get("authors"), "authors"):
        name = " ".join(clean_text(name).split())
        if name and name not in authors:
            authors.append(name)
    if not authors:
        raise InvalidRecord("Author names are missing after cleaning.")

    year = record.get("first_publish_year")
    if year is not None and (isinstance(year, bool) or not isinstance(year, int) or year <= 0):
        raise InvalidRecord("first_publish_year must be a positive integer or null.")

    source_url = record.get("source_url")
    if work_id.startswith("USER-"):  # Added through the API, so there is no Open Library page
        expected_url = None
        if source_url is not None:
            raise InvalidRecord("source_url must be null for user-added books.")
    else:
        expected_url = f"https://openlibrary.org/works/{work_id}"
        if not isinstance(source_url, str) or source_url.strip().rstrip("/") != expected_url:
            raise InvalidRecord("source_url must match the Open Library work ID.")

    cleaned = dict(record)
    cleaned.update(
        work_id=work_id,
        title=title,
        authors=authors,
        description=description,
        author_bio=clean_text(record.get("author_bio")) or None,
        subjects=normalize_subjects(record.get("subjects")),
        first_publish_year=year,
        source_url=expected_url,
    )
    return cleaned


def clean_records(records: Iterable[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Keep the first valid record per work and report all skipped records."""
    cleaned_records: list[dict[str, Any]] = []
    seen: set[str] = set()
    reasons: Counter[str] = Counter()
    rejected: list[dict[str, Any]] = []
    input_count = 0
    removed_subject_count = 0

    for position, record in enumerate(records, 1):
        input_count += 1
        try:
            cleaned = clean_record(record)
            if cleaned["work_id"] in seen:
                raise InvalidRecord("Duplicate work_id.")
        except InvalidRecord as error:
            reason = str(error)
            reasons[reason] += 1
            rejected.append({"record_number": position, "work_id": record.get("work_id") if isinstance(record, dict) else None, "reason": reason})
            continue
        seen.add(cleaned["work_id"])
        removed_subject_count += len(record["subjects"]) - len(cleaned["subjects"])
        cleaned_records.append(cleaned)

    report = {
        "stage": "cleaning_only",
        "input_records": input_count,
        "output_records": len(cleaned_records),
        "dropped_records": input_count - len(cleaned_records),
        "dropped_percent": round(100 * (input_count - len(cleaned_records)) / input_count, 2) if input_count else 0,
        "drop_reasons": dict(reasons),
        "rejected_records": rejected,
        "missing_publication_year": sum(record["first_publish_year"] is None for record in cleaned_records),
        "missing_author_bio": sum(record["author_bio"] is None for record in cleaned_records),
        "removed_subject_entries": removed_subject_count,
        "notes": [
            "No chunking, token counting, language detection, or fact verification performed.",
            "Unknown publication years and biographies remain null.",
            "Short nonempty descriptions are retained pending the team's quality threshold.",
            "Author biographies are preserved without assigning them to individual coauthors.",
        ],
    }
    return cleaned_records, report
