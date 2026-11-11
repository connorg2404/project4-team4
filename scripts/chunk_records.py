"""Run from the repository root: python -m scripts.chunk_records --input PATH."""

import argparse
import hashlib
import json
from pathlib import Path

from app.services.chunking_service import build_chunks, load_tokenizer
from scripts.clean_records import read_jsonl


def main():
    parser = argparse.ArgumentParser(description="Clean and chunk books using the pinned Nomic tokenizer.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if len({p.resolve() for p in (args.input, args.output, args.report)}) != 3:
        parser.error("Input, output and report must be different files.")
    if args.output.exists() or args.report.exists():
        parser.error("Output/report already exists; choose new filenames.")
    try:
        records = list(read_jsonl(args.input))
        if not records:
            parser.error("Input contains no records. The repository's initial works.jsonl may be empty.")
        tokenizer = load_tokenizer()
        chunks, report = build_chunks(records, tokenizer)
        if not chunks:
            parser.error("No chunks produced; inspect input data before indexing.")
        report["input_sha256"] = hashlib.sha256(args.input.read_bytes()).hexdigest()
        for path in (args.output, args.report):
            path.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in chunks), encoding="utf-8")
        report["output_sha256"] = hashlib.sha256(args.output.read_bytes()).hexdigest()
        args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(f"Books with chunks: {report['works_with_chunks']} | Chunks: {len(chunks)}")
    print(f"Maximum tokens including context/prefix: {report['token_length']['max']}")
    print(f"Issues to review: {len(report['issues'])} | Short chunks: {len(report['short_body_chunk_ids'])}")
    print(f"Output: {args.output}\nReport: {args.report}")
    print("These are chunks for John to embed, not embedding vectors.")


if __name__ == "__main__":
    main()
