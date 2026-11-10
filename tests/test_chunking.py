import copy
import unittest

from app.embedding_config import DOCUMENT_PREFIX, EMBEDDING_MODEL, EMBEDDING_REVISION
from app.services.chunking_service import build_chunks, load_tokenizer, token_count


def book():
    return {
        "work_id": "OL123W", "title": "A Space Journey", "authors": ["Jane Author"],
        "description": "<p>A scientist travels through space and discovers a new world.</p>",
        "subjects": ["Science Fiction", "New York Times Bestseller", "nyt:chart=2026"],
        "author_bio": "Jane writes science fiction.", "author_bio_author": "Jane Author",
        "first_publish_year": None, "source_url": "https://openlibrary.org/works/OL123W",
    }


class ChunkingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = load_tokenizer()

    def test_nine_field_schema_and_source_metadata(self):
        chunks, report = build_chunks([book()], self.tokenizer)
        fields = {"chunk_id", "text", "work_id", "title", "authors", "subjects", "section", "first_publish_year", "source_url"}
        self.assertEqual(len(chunks), 3)
        for chunk in chunks:
            self.assertEqual(set(chunk), fields)
            self.assertIsNone(chunk["first_publish_year"])
            self.assertEqual(chunk["source_url"], book()["source_url"])
            self.assertIn("Title: A Space Journey", chunk["text"])
            self.assertNotIn("<p>", chunk["text"])
            self.assertFalse(chunk["text"].startswith(DOCUMENT_PREFIX))
        self.assertEqual(report["embedding_model"], EMBEDDING_MODEL)
        self.assertEqual(report["embedding_revision"], EMBEDDING_REVISION)

    def test_cap_counts_prefix_header_and_special_tokens(self):
        record = book()
        record["description"] = " ".join(f"Scientist number {i} explores a new planet." for i in range(120))
        chunks, _ = build_chunks([record], self.tokenizer)
        descriptions = [c for c in chunks if c["section"] == "description"]
        self.assertGreater(len(descriptions), 1)
        for chunk in chunks:
            self.assertLessEqual(token_count(self.tokenizer, chunk["text"]), 400)
        text = " ".join(c["text"] for c in descriptions)
        for i in range(120):
            self.assertIn(f"Scientist number {i} explores a new planet.", text)

    def test_unknown_bio_owner_is_reported_without_dropping_book(self):
        record = book()
        record.pop("author_bio_author")
        chunks, report = build_chunks([record], self.tokenizer)
        self.assertFalse(any(c["section"] == "author_bio" for c in chunks))
        self.assertEqual(report["works_with_chunks"], 1)
        self.assertTrue(report["issues"])

    def test_mismatched_bio_owner_is_not_assigned_to_first_author(self):
        record = book()
        record["author_bio_author"] = "Somebody Else"
        chunks, report = build_chunks([record], self.tokenizer)
        self.assertFalse(any(c["section"] == "author_bio" for c in chunks))
        self.assertEqual(len(report["issues"]), 1)

    def test_second_author_is_named_in_bio_chunk(self):
        record = book()
        record["authors"] = ["First Author", "Jane Author"]
        chunks, _ = build_chunks([record], self.tokenizer)
        bio = next(c for c in chunks if c["section"] == "author_bio")
        self.assertIn("Biography of Jane Author:", bio["text"])

    def test_indivisible_oversized_sentence_is_reported(self):
        record = book()
        record["description"] = "space " * 600 + "."
        chunks, report = build_chunks([record], self.tokenizer)
        self.assertFalse(any(c["section"] == "description" for c in chunks))
        self.assertTrue(any("omitted_text" in issue for issue in report["issues"]))
        self.assertTrue(all(token_count(self.tokenizer, c["text"]) <= 400 for c in chunks))

    def test_subject_lists_are_split_without_losing_labels(self):
        record = book()
        record["subjects"] = [f"subject number {i}" for i in range(160)]
        chunks, _ = build_chunks([record], self.tokenizer)
        subject_text = " ".join(c["text"] for c in chunks if c["section"] == "subjects")
        for subject in record["subjects"]:
            self.assertIn(subject, subject_text)
        self.assertTrue(all(token_count(self.tokenizer, c["text"]) <= 400 for c in chunks))

    def test_missing_sections_do_not_produce_empty_chunks(self):
        record = book()
        record["subjects"] = []
        record["author_bio"] = None
        chunks, _ = build_chunks([record], self.tokenizer)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["section"], "description")

    def test_duplicate_work_ids_do_not_duplicate_chunks(self):
        chunks, report = build_chunks([book(), book()], self.tokenizer)
        self.assertEqual(len(chunks), 3)
        self.assertEqual(report["cleaning"]["dropped_records"], 1)
        self.assertEqual(len({c["chunk_id"] for c in chunks}), len(chunks))

    def test_deterministic_and_does_not_mutate_input(self):
        original = book()
        before = copy.deepcopy(original)
        first = build_chunks([original], self.tokenizer)
        self.assertEqual(first, build_chunks([original], self.tokenizer))
        self.assertEqual(original, before)

    def test_bestseller_label_is_kept_without_tracking_tag(self):
        chunks, _ = build_chunks([book()], self.tokenizer)
        for chunk in chunks:
            self.assertIn("new york times bestseller", chunk["subjects"])
            self.assertFalse(any(s.startswith("nyt:") for s in chunk["subjects"]))

    def test_empty_input_and_invalid_cap(self):
        chunks, report = build_chunks([], self.tokenizer)
        self.assertEqual(chunks, [])
        self.assertEqual(report["token_length"]["max"], 0)
        with self.assertRaises(ValueError):
            build_chunks([book()], self.tokenizer, max_tokens=20)


if __name__ == "__main__":
    unittest.main()
