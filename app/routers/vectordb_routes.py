import json
from collections.abc import Iterator
from typing import Literal

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.services.langchain_service import LLM_MODEL_ID, answer_question, stream_answer
from app.services.vectordb_service import collection_count

# These paths sit at the root (/health, /ask, /ask/stream) because that's
# what the Streamlit frontend and evaluation/run_eval.py call.
router = APIRouter(tags=["librarian"])


class HistoryMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=5000)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    top_k: int = Field(5, ge=1, le=20)
    user_books: list[str] = Field(default_factory=list, max_length=50)  # Books the user already owns
    history: list[HistoryMessage] = Field(default_factory=list, max_length=20)  # Earlier turns, oldest first

    def history_dicts(self) -> list[dict[str, str]]:
        return [message.model_dump() for message in self.history]


# NOTE: these are plain `def`, not `async def`. Embedding and Bedrock calls block,
# and FastAPI runs plain `def` endpoints in a thread pool so other requests aren't frozen.

@router.get("/health")
def health():
    try:
        chunks = collection_count()
    except ValueError as error:  # e.g. the index was built with a different embedding model
        raise HTTPException(status_code=503, detail=str(error))
    if chunks == 0:
        raise HTTPException(status_code=503, detail="Vector DB is empty. Run scripts.index_chunks first.")
    return {"status": "ok", "model": LLM_MODEL_ID, "index_size": chunks}  # Field names the frontend reads


@router.post("/ask")
def ask(request: AskRequest):
    return answer_question(request.question, request.top_k, request.user_books, request.history_dicts())


def _sse(data: dict, event: str | None = None) -> str:
    """One server-sent event. JSON keeps newlines inside a token from breaking the format."""
    prefix = f"event: {event}\n" if event else ""
    return f"{prefix}data: {json.dumps(data, ensure_ascii=False)}\n\n"


def _events(request: AskRequest) -> Iterator[str]:
    try:
        for item in stream_answer(request.question, request.top_k, request.user_books, request.history_dicts()):
            if isinstance(item, dict):  # The final summary: sources, refused, latency_ms
                yield _sse(item, event="sources")
            else:
                yield _sse({"token": item})
        yield _sse({}, event="done")
    except Exception as error:  # Tell the client instead of silently dropping the connection
        message = " ".join(f"{type(error).__name__}: {error}".split())
        yield f"event: error\ndata: {message}\n\n"


@router.post("/ask/stream")
def ask_stream(request: AskRequest):
    return StreamingResponse(
        _events(request),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
