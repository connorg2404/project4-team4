"""Run from the project root: python -m scripts.clean_records --input PATH."""

import argparse
import json
from pathlib import Path

from app.services.cleaning_service import clean_records


def read_jsonl(path: Path):
    with path.open(encoding="utf-8-sig") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at line {line_number}: {error.msg}") from error


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean Open Library records without guessing missing facts.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/cleaned/cleaned_50.jsonl"))
    parser.add_argument("--report", type=Path, default=Path("data/cleaned/cleaning_report.json"))
    args = parser.parse_args()
    resolved = [path.resolve() for path in (args.input, args.output, args.report)]
    if len(set(resolved)) != 3:
        parser.error("Input, output, and report must be different files.")
    if args.output.exists() or args.report.exists():
        parser.error("Output/report already exists. Choose new filenames to preserve the previous run.")
    try:
        records, report = clean_records(read_jsonl(args.input))
    except (OSError, ValueError) as error:
        parser.error(str(error))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as target:
        for record in records:
            target.write(json.dumps(record, ensure_ascii=False) + "\n")
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Input: {report['input_records']} | Kept: {report['output_records']} | Dropped: {report['dropped_records']}")
    print(f"Unknown publication years: {report['missing_publication_year']}")
    print(f"Missing author biographies: {report['missing_author_bio']}")
    print(f"Cleaned records: {args.output}")
    print(f"Report: {args.report}")
    print("Cleaning only: these records are not yet chunks for indexing.")


if __name__ == "__main__":
    main()
