# Librarian Bot (Project 4 – Team 4)

A book-recommendation chatbot built on **Retrieval-Augmented Generation (RAG)**. Ask for a book by topic, mood or theme and the bot searches a catalog of 800 highly rated works from [Open Library](https://openlibrary.org/developers/dumps), then has an LLM answer using *only* the books it retrieved, with sources cited. Questions that aren't about books are refused, instead of hallucinated.

## How It Works

```
Open Library dumps ──► digester.py ──► clean ──► chunk ──► embed + index (Chroma)
                                                                │
Streamlit UI ──► FastAPI (/ask, /ask/stream) ──► retrieve top-k books ──► LLM answer + sources
```

1. **Ingest** – `digester.py` streams the Open Library dumps and picks top-rated works per category into `app/data/raw/works.jsonl`.
2. **Clean and chunk** – records are cleaned, then split into description, subjects and author-bio chunks (≤ 400 tokens, measured with the Nomic tokenizer). The current dataset has 800 works and 2,506 chunks.
3. **Embed and index** – chunks are embedded with `nomic-ai/nomic-embed-text-v1.5` (pinned revision, 768 dimensions) and stored in a local **ChromaDB** collection. The index records which model built it, so it is never queried with a different one.
4. **Retrieve** – the question is embedded, the closest chunks are grouped into distinct books, and books the user already owns are filtered out.
5. **Answer** – if the best match is close enough, the book list goes straight to the LLM (Amazon Nova Lite on AWS Bedrock via LangChain). Otherwise a gate chain decides whether the message is a book question, small talk, or something to refuse.

## Features

- Book recommendations grounded in the catalog, with cited sources and links
- Streaming answers over server-sent events
- "Books I own" list so the bot recommends something new
- Short conversation history for follow-up questions
- Refusal of off-topic questions (distance threshold plus an LLM gate)
- Endpoint to add new books to the catalog at runtime
- Evaluation harness measuring hit@k, precision@k, refusal accuracy, false refusals and latency

## Tech Stack

Python 3.12 · FastAPI · Streamlit · LangChain · AWS Bedrock (Amazon Nova Lite) · ChromaDB · Sentence-Transformers (Nomic Embed v1.5) · Pydantic · pytest

## Project Structure

```
project4-team4/
├── digester.py                  # Pulls top works from Open Library dumps
├── scripts/
│   ├── clean_records.py         # Clean raw records
│   ├── chunk_records.py         # Chunk records with the Nomic tokenizer
│   ├── fill_years.py            # Fill missing first_publish_year values
│   └── index_chunks.py          # Embed chunks and upsert into Chroma
├── app/
│   ├── app.py                   # FastAPI entry point
│   ├── routers/                 # /books/* and /health, /ask, /ask/stream
│   ├── services/                # chunking, cleaning, ingest, LangChain RAG, vector DB
│   ├── pages/frontend.py        # Streamlit chat UI
│   ├── embedding_config.py      # Pinned embedding model and prefixes
│   ├── chroma_db/               # Persisted vector index
│   ├── data/raw/works.jsonl     # Raw catalog
│   └── evaluation/              # test_set.csv, build_test_set.py, run_eval.py
├── data/chunks/                 # Chunked output and chunk report
└── tests/                       # Cleaning and chunking tests
```

## Getting Started

### Prerequisites

- Python 3.12
- AWS credentials with access to Amazon Nova Lite on Bedrock in `us-east-2`
- Python packages for the app: `fastapi`, `uvicorn`, `streamlit`, `requests`, `pydantic`, `langchain-core`, `langchain-aws`, `chromadb`, `sentence-transformers`. The root `requirements.txt` is currently empty, and `requirements-chunking.txt` only covers the chunking step (`transformers`, `pysbd`).

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-chunking.txt
pip install fastapi uvicorn streamlit requests pydantic langchain-core langchain-aws chromadb sentence-transformers
```

### Run the API

From the project root:

```bash
python -m uvicorn app.app:app --reload
```

The API starts on `http://localhost:8000` (interactive docs at `/docs`). The embedding model loads at startup, so the first launch downloads it.

### Run the frontend

In a second terminal, from the project root:

```bash
streamlit run app/streamlit.py
```

Set `LIBRARIAN_API_URL` if the API isn't on `http://localhost:8000`.

### Rebuild the index (optional)

A built index is included in `app/chroma_db/`. To regenerate it from the raw catalog:

```bash
python -m scripts.chunk_records --input app/data/raw/works.jsonl \
    --output data/chunks/chunks.jsonl --report data/chunks/chunk_report.json
python -m scripts.index_chunks --input data/chunks/chunks.jsonl
```

To refresh the raw catalog itself, download the Open Library works, authors and ratings dumps into `data/dumps/` and run `python digester.py`.

## API Reference

| Method | Endpoint | Description |
| --- | --- | --- |
| GET | `/health` | Status, LLM model, and number of indexed chunks |
| POST | `/ask` | Ask a question; returns `answer`, `sources`, `refused`, `latency_ms` |
| POST | `/ask/stream` | Same request, streamed as server-sent events (token chunks, then a `sources` event) |
| POST | `/books/rag_chat` | Simple RAG chat with `{"input": "..."}` |
| POST | `/books/ingest` | Add a new book to the catalog |
| POST | `/books/chat` | General chat with no retrieval |

Example:

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Suggest a hopeful sci-fi novel about space", "top_k": 5}'
```

## Evaluation and Tests

```bash
python app/evaluation/run_eval.py --label "baseline"   # needs the API running
python app/evaluation/run_eval.py --stream             # also measures time to first token
python -m pytest tests/
```

The eval script scores hit@k (team target: 80%), precision@k, refusal accuracy (target: 100%), false refusals and latency, and writes results to `app/evaluation/results/`. Tune on the default split; use `--split holdout` only for the final check.

## Contributors

Thanks to the team members who built this project:

| Contributor | GitHub |
| --- | --- |
| John Chen | [@Jc3649124817](https://github.com/Jc3649124817) |
| Abishek Shankara | [@AbishekSh](https://github.com/AbishekSh) |
| Connor Townson | [@connorg2404](https://github.com/connorg2404) |
| Benedict Martinez | [@Rymarmar](https://github.com/Rymarmar) |
| Nigel Sequeira | [@nrsequeira](https://github.com/nrsequeira) |
