
# Implements the shared contract with canned data: /ask, /ask/stream, /health.
# The source_url values are placeholders; the books and summaries are not real bestseller data.

# Run:  python -m uvicorn mock_api:app --port 8000
# Try:  "books about monsters" (answers), "what's the weather?" (refused),
#       "recommend something" (skips any books in the request's user_books)

import asyncio
import json
import time

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

app = FastAPI(title="Mock Librarian Bot API")

CATALOG = [
    {
        "title": "Frankenstein",
        "authors": ["Mary Shelley"],
        "source_url": "https://example.com/mock-bestsellers#frankenstein",
        "keywords": {"frankenstein", "shelley", "monster", "monsters", "gothic", "horror", "science"},
        "summary": "a young scientist builds a living creature and then recoils from what he has made",
    },
    {
        "title": "Pride and Prejudice",
        "authors": ["Jane Austen"],
        "source_url": "https://example.com/mock-bestsellers#pride-and-prejudice",
        "keywords": {"pride", "prejudice", "austen", "romance", "regency", "marriage", "manners"},
        "summary": "Elizabeth Bennet and Mr. Darcy navigate first impressions, class and family pressure",
    },
    {
        "title": "Dune",
        "authors": ["Frank Herbert"],
        "source_url": "https://example.com/mock-bestsellers#dune",
        "keywords": {"dune", "herbert", "desert", "science", "fiction", "scifi", "politics", "planet"},
        "summary": "a noble family takes control of a desert planet that is the only source of a vital spice",
    },
    {
        "title": "Neuromancer",
        "authors": ["William Gibson"],
        "source_url": "https://example.com/mock-bestsellers#neuromancer",
        "keywords": {"neuromancer", "gibson", "cyberpunk", "ai", "hacker", "science", "fiction", "scifi"},
        "summary": "a washed-up hacker is hired for one last job involving a powerful artificial intelligence",
    },
]
THRESHOLD = 0.15
RECOMMEND_WORDS = {"recommend", "recommendation", "recommendations", "suggest", "next"}
REFUSAL = "I don't know. I couldn't find anything relevant in the catalog."

class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int = Field(default=4, ge=1, le=20)
    user_books: list[str] = Field(default_factory=list, max_length=100)

def words_of(text: str) -> set[str]:
    return {w.strip("?.,!'\"").lower() for w in text.split()}

def retrieve(question: str, top_k: int):
    words = words_of(question)
    scored = []
    for book in CATALOG:
        overlap = len(words & book["keywords"])
        score = overlap / 3  # crude stand-in for a similarity score
        if score >= THRESHOLD:
            scored.append((min(score, 0.99), book))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return scored[:top_k]

def recommend(top_k: int, user_books: list[str]):
    owned = [b.lower() for b in user_books]
    pool = [b for b in CATALOG if not any(b["title"].lower() in o for o in owned)]
    return [(0.5, b) for b in pool[:top_k]]

def build_response(question: str, top_k: int, user_books: list[str]) -> dict:
    started = time.perf_counter()
    wants_recommendation = bool(words_of(question) & RECOMMEND_WORDS)
    hits = recommend(top_k, user_books) if wants_recommendation else retrieve(question, top_k)
    if not hits:
        return {
            "answer": REFUSAL,
            "sources": [],
            "refused": True,
            "latency_ms": int((time.perf_counter() - started) * 1000),
        }
    lines = " ".join(f"{b['title']} by {', '.join(b['authors'])}: {b['summary']}." for _, b in hits)
    if wants_recommendation:
        intro = (
            f"Skipping the {len(user_books)} book(s) on your list, you might like: "
            if user_books
            else "You haven't added any books yet, so here are some popular picks: "
        )
        answer = intro + lines
    else:
        answer = lines
    sources = [
        {"title": b["title"], "authors": b["authors"], "source_url": b["source_url"], "score": round(s, 3)}
        for s, b in hits
    ]
    return {
        "answer": answer,
        "sources": sources,
        "refused": False,
        "latency_ms": int((time.perf_counter() - started) * 1000) + 300,
    }

@app.post("/ask")
async def ask(req: AskRequest):
    await asyncio.sleep(0.5)  # pretend the model is thinking
    return build_response(req.question, req.top_k, req.user_books)

@app.post("/ask/stream")
async def ask_stream(req: AskRequest):
    result = build_response(req.question, req.top_k, req.user_books)

    async def events():
        await asyncio.sleep(0.5)
        words = result["answer"].split(" ")
        for i, word in enumerate(words):
            token = word if i == 0 else " " + word
            yield f"data: {json.dumps({'token': token})}\n\n"
            await asyncio.sleep(0.04)
        final = {k: result[k] for k in ("sources", "refused", "latency_ms")}
        yield f"event: sources\ndata: {json.dumps(final)}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")

@app.get("/health")
async def health():
    return {"status": "ok", "model": "mock", "index_size": len(CATALOG)}