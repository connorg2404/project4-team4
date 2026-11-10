# Book chunking handoff

Person 2 produces chunks; Person 3 generates embeddings, builds the vector index,
and implements retrieval. These files do not change the API, UI, or LLM chain.

## Model contract

- Model: `nomic-ai/nomic-embed-text-v1.5`
- Pinned model/tokenizer revision: `e9b6763023c676ca8431644204f50c2b100d9aab`
- Full embedding size for Person 3: 768 dimensions; normalize consistently for the chosen similarity metric.
- Person 3 loads this model locally with Sentence-Transformers, as required by the project.
- Person 2 loads ONLY the corresponding tokenizer; no PyTorch/model weights needed to generate chunks.
- Person 3 prepends `search_document: ` to chunk text exactly once when embedding documents.
- Person 3 prepends `search_query: ` to a question exactly once when embedding queries.
- The model, revision, dimensions, and preprocessing must match across indexing and search.
- `app/embedding_config.py` is the shared source for those values.
- Nova Lite is the separate answer-generation model, not used in this chunk builder.

The `text` field is unprefixed, readable content. The cap counts the eventual
document prefix, context header, and tokenizer special tokens. Default cap: 400.
Target 200-400 where sufficient source content exists; short source sections are
not padded with fabricated text.

## Files and execution

From the repository root, using Python 3.12:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-chunking.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -v
.\.venv\Scripts\python.exe -m scripts.chunk_records --input "C:\Users\benma\Downloads\works.jsonl" --output data/chunks/works.nomic-v1_5.jsonl --report data/chunks/works.nomic-v1_5.report.json
```

Use another input path on another machine. The first run downloads tokenizer
files. An existing output/report is never overwritten: select fresh filenames.
The builder accepts raw records and runs the existing cleaner internally.

If Transformers reports that PyTorch is missing, that is expected in this
tokenizer-only environment. It does not mean chunk generation failed. Person 3's
Sentence-Transformers environment will need its own model runtime dependencies.

## Input and output contract

Required raw fields follow the existing cleaner. `author_bio_author` must name a
listed author before a biography is chunked. An unmatched/missing owner causes
only the biography section to be omitted and reported; it does not discard the
book's description or subjects.

Output JSONL has exactly these nine fields:

`chunk_id`, `text`, `work_id`, `title`, `authors`, `subjects`, `section`,
`first_publish_year`, `source_url`.

- IDs: `work_id::section::index`, deterministic for identical inputs/config.
- Sections: description, subjects, author_bio. No editions are invented.
- Description/biography chunks carry title, authors, and a section label.
- Biography labels explicitly name the biography's owner.
- Unknown years remain JSON null. Lists remain lists in JSONL.
- Person 3 must adapt lists/nulls if the chosen vector database requires scalar
  metadata, and reconstruct lists for the API response.
- Changed data/config can change chunk ordinals; Person 3 must reconcile stale
  IDs when reindexing instead of only appending vectors.

## Splitting and known limits

Paragraphs remain intact when they fit. Oversized paragraphs split at sentence
boundaries using an English abbreviation-aware segmenter. Subject tags are
complete units. Small trailing fragments are merged/rebalanced within a section
where the cap allows. No overlap is added in this initial version.

An indivisible unit above the cap is omitted and recorded verbatim in the report,
not silently truncated. No such omission occurred on the verified 800-book input.
New data can trigger this condition; review `issues` before handing it to the indexer.

Short standalone sections are retained and reported. Cross-section merging and
an English-only/language-tagging policy remain review tasks. The current builder
preserves source languages and uses English sentence segmentation; it does not
claim to detect language. Source boilerplate is not comprehensively classified.
Reference-style Markdown links are reduced to labels and reference-definition
URLs removed. Readable bestseller labels remain source tags, not verified current
rankings or sales claims. Publication years are not independently verified.

## Verified first run (2026-11-10)

- Baseline: main commit `9d60d55`.
- Input: updated Downloads `works.jsonl`, 800 records.
- Input SHA-256: `bacd2a1fffb3fd59074550692317efe6da8be7cb6a1390a9ec33d4398efe58ed`.
- 38 tests passed: 26 existing cleaning tests plus 12 chunking tests.
- 2,506 chunks covering all 800 works: 850 description, 836 subjects, 820 author_bio.
- Token lengths including prefix/context/special tokens: min 18, median 147, max 400.
- 1,642 chunks under 200 tokens; 864 from 200 through 400; zero above 400.
- 418 chunks have body text under 50 tokens and are flagged for review.
- Zero omission issues, duplicate chunk IDs, or detected HTML tags.
- Output SHA-256: `592f01b5d110d6d43eea394ee89241676c63ef867eb8be8f2cff2cc034650bab`.
- These checks do not establish retrieval precision, refusal accuracy, or end-to-end API correctness.

Person 1 should review the provided deterministic sample of 30 chunks. Person 3
can start indexing this initial artifact while that quality review proceeds.

## Existing integration issues to report to owners

At `9d60d55`, `app/data/raw/works.jsonl` is empty. Use the actual downloaded
800-record dataset, and agree on a shared data location. The tracked
`app/evaluation/test_set.csv` contains Python source rather than generated CSV;
Person 5 should regenerate it from the real dataset. The backend still exposes
`/books/chat` while the frontend expects `/ask`, `/ask/stream`, and `/health`.
Embedding/database code and Bedrock integration are not present in this baseline.
