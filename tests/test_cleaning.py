import copy
import unittest

from app.services.cleaning_service import InvalidRecord, clean_record, clean_records, clean_text, normalize_subjects


def example_record():
    return {
        "work_id": "OL123W",
        "title": "  Example Book  ",
        "authors": ["Example   Author"],
        "author_bio": None,
        "description": "<p>A book about <b>space</b>.</p><p>It explores **discovery**.</p>",
        "subjects": ["Science Fiction", "science fiction", "nyt:example=2026"],
        "first_publish_year": None,
        "source_url": "https://openlibrary.org/works/OL123W",
        "popularity": 12,
        "cover_id": 456,
    }


class CleaningTests(unittest.TestCase):
    def test_html_removal_preserves_text_and_paragraphs(self):
        self.assertEqual(clean_text("<p>First <b>idea</b>.</p><p>Second idea.</p>"), "First idea.\n\nSecond idea.")

    def test_hidden_html_is_removed(self):
        self.assertEqual(clean_text("Visible<script>hidden()</script><style>hidden</style> text"), "Visible text")

    def test_markdown_keeps_link_label_and_emphasis_text(self):
        self.assertEqual(clean_text("**Read** [the guide](https://example.com) and *learn*."), "Read the guide and learn.")

    def test_whitespace_entities_and_controls(self):
        self.assertEqual(clean_text("A\t &amp;\u00a0 B\x00\r\n\r\nC"), "A & B\n\nC")

    def test_subjects_keep_meaning_and_remove_tracking_duplicates(self):
        self.assertEqual(normalize_subjects(["Fantasy", " fantasy ", "Romance languages", "nyt:chart=2026"]), ["fantasy", "romance languages"])

    def test_record_metadata_and_unknown_year_are_preserved(self):
        result = clean_record(example_record())
        self.assertIsNone(result["first_publish_year"])
        self.assertEqual(result["work_id"], "OL123W")
        self.assertEqual(result["title"], "Example Book")
        self.assertEqual(result["authors"], ["Example Author"])
        self.assertEqual(result["source_url"], "https://openlibrary.org/works/OL123W")
        self.assertEqual(result["cover_id"], 456)
        self.assertEqual(result["popularity"], 12)

    def test_known_year_is_preserved_without_claiming_verification(self):
        record = example_record()
        record["first_publish_year"] = 1979
        self.assertEqual(clean_record(record)["first_publish_year"], 1979)

    def test_missing_bio_stays_null(self):
        self.assertIsNone(clean_record(example_record())["author_bio"])

    def test_empty_description_is_rejected(self):
        record = example_record()
        record["description"] = "<p> </p>"
        with self.assertRaisesRegex(InvalidRecord, "Description"):
            clean_record(record)

    def test_empty_title_is_rejected(self):
        record = example_record()
        record["title"] = " "
        with self.assertRaisesRegex(InvalidRecord, "Title"):
            clean_record(record)

    def test_invalid_field_types_are_rejected(self):
        for field, value in (("authors", "Author"), ("subjects", [1]), ("first_publish_year", True), ("description", 1)):
            with self.subTest(field=field):
                record = example_record()
                record[field] = value
                with self.assertRaises(InvalidRecord):
                    clean_record(record)

    def test_source_must_match_work(self):
        record = example_record()
        record["source_url"] = "https://openlibrary.org/works/OL999W"
        with self.assertRaisesRegex(InvalidRecord, "source_url"):
            clean_record(record)

    def test_duplicate_work_is_reported(self):
        output, report = clean_records([example_record(), example_record()])
        self.assertEqual(len(output), 1)
        self.assertEqual(report["dropped_records"], 1)
        self.assertEqual(report["drop_reasons"], {"Duplicate work_id.": 1})

    def test_rejected_record_does_not_hide_later_valid_record(self):
        invalid = example_record()
        invalid["description"] = None
        output, report = clean_records([invalid, example_record()])
        self.assertEqual(len(output), 1)
        self.assertEqual(report["dropped_records"], 1)

    def test_cleaning_does_not_mutate_input(self):
        record = example_record()
        original = copy.deepcopy(record)
        clean_record(record)
        self.assertEqual(record, original)

    def test_repeat_cleaning_is_stable(self):
        once = clean_record(example_record())
        self.assertEqual(clean_record(once), once)

    def test_unknown_years_are_not_dropped(self):
        output, report = clean_records([example_record()])
        self.assertEqual(len(output), 1)
        self.assertEqual(report["missing_publication_year"], 1)
        self.assertEqual(report["missing_author_bio"], 1)

    def test_empty_input_report(self):
        output, report = clean_records([])
        self.assertEqual(output, [])
        self.assertEqual(report["dropped_percent"], 0)


class ReviewFeedbackTests(unittest.TestCase):
    def test_standalone_separator_styles_are_removed(self):
        for rule in ("---", "----------", "***", "___", "- - -", "* * *", "_ _ _"):
            with self.subTest(rule=rule):
                self.assertEqual(
                    clean_text(f"First paragraph.\n  {rule}  \nSecond paragraph."),
                    "First paragraph.\n\nSecond paragraph.",
                )

    def test_windows_line_endings_around_separators(self):
        self.assertEqual(
            clean_text("First.\r\n\r\n----------\r\n\r\nSecond."),
            "First.\n\nSecond.",
        )

    def test_normal_hyphens_and_attribution_are_preserved(self):
        text = "Jean-Paul wrote science-fiction in 1990-2000.\n---From Wikipedia"
        self.assertEqual(clean_text(text), text)

    def test_separator_cleanup_applies_to_bio_and_description(self):
        record = example_record()
        record["description"] = "A space adventure.\n---\nExplores discovery."
        record["author_bio"] = "An author.\n----------\nWrites about space."
        record["author_bio_author"] = "Example Author"
        result = clean_record(record)
        self.assertEqual(result["description"], "A space adventure.\n\nExplores discovery.")
        self.assertEqual(result["author_bio"], "An author.\n\nWrites about space.")
        self.assertEqual(result["author_bio_author"], "Example Author")

    def test_separator_only_bio_becomes_missing(self):
        record = example_record()
        record["author_bio"] = "----------"
        self.assertIsNone(clean_record(record)["author_bio"])

    def test_separator_only_description_is_rejected(self):
        record = example_record()
        record["description"] = "---\n***\n___"
        with self.assertRaisesRegex(InvalidRecord, "Description"):
            clean_record(record)

    def test_bestseller_labels_survive_but_tracking_tags_do_not(self):
        self.assertEqual(
            normalize_subjects([
                "New York Times Bestseller",
                "New york times bestseller",
                "nyt:combined-print-and-e-book-fiction=2020-01-01",
                "Fantasy",
            ]),
            ["new york times bestseller", "fantasy"],
        )

    def test_separator_cleanup_is_stable(self):
        once = clean_text("**An author.**\n----------\nMore details.")
        self.assertEqual(clean_text(once), once)


if __name__ == "__main__":
    unittest.main()
