import os
import re
import time
from collections.abc import Iterator
from difflib import SequenceMatcher
from typing import Any

from langchain_aws import ChatBedrockConverse
from langchain_classic.chains.conversation.base import ConversationChain
from langchain_classic.memory import ConversationBufferWindowMemory, ConversationSummaryMemory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app.services.vectordb_service import search_collection

LLM_MODEL_ID = "amazon.nova-lite-v1:0"  # The Bedrock model ID we have access to
LLM_REGION = "us-east-2"                # The region where we have access to that model

llm = ChatBedrockConverse(
    model=LLM_MODEL_ID,
    region_name=LLM_REGION,
    temperature=0.9,
)

# Lower temperature for RAG: we want it to stick to the catalog, not get creative
rag_llm = ChatBedrockConverse(
    model=LLM_MODEL_ID,
    region_name=LLM_REGION,
    temperature=0.3,
)

# Define a general prompt to help us define the LLM's persona
# We define system instructions, then pass along the user's input
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a general chatbot that answers questions. You give helpful and concise advice"),
    ("user", "{input}")
])

# Basic general chain that we'll call in the router for chat functionality
def get_general_chain():

    # return a chain! This syntax is called "LCEL" LangChain Expression Language
        # Pass in the prompt
        # Pass in the LLM
        # This will get returned to the router to get invoked
    return prompt | llm


# Create a different chain that has conversational memory
# This is gonna be very taxing on our local LLMs
def get_memory_chain():

    # Create a memory object

    # Window memory - remembers the last "k" interactions
    ### memory = ConversationBufferWindowMemory(k=3)

    # Summary memory - keeps a running paragraph of what we've talked about
    memory = ConversationSummaryMemory(llm=llm)
    # Our LLM is doing the summarization THEN generating a response... expensive

    # Similar prompt, but with history attached
    # These older memory classes expect a {history} prompt variable
    memory_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a general chatbot that answers questions. You give helpful and concise advice"),
        ("user", "Conversation History: {history}"
                "Current Input: {input}")
    ])

    # Make and return a chain in the old fashioned way (NOT using LCEL)
    return ConversationChain(
        llm = llm,
        memory = memory,
        prompt = memory_prompt
    )


# ---------------------------------------------------------------------------
# RAG: retrieve books from the vector DB, then have the LLM answer from them
# ---------------------------------------------------------------------------

# If even the closest chunk is further than this (cosine distance, 0 = identical),
# we refuse instead of asking the LLM. TUNE THIS on the eval's tune split, not holdout.
REFUSAL_DISTANCE = float(os.getenv("REFUSAL_DISTANCE", "0.55"))

# Each book has ~3 chunks (description, subjects, author bio), so fetch extra
# chunks to still end up with top_k distinct books after grouping and filtering.
CHUNKS_PER_BOOK = 4

REFUSAL_MESSAGE = (
    "I couldn't find anything in the catalog that matches that question. "
    "Try asking about a book, author, genre, or theme."
)

rag_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a librarian for a small book catalog. The CATALOG below is the ONLY set of books that exist.

Rules:
- Only mention books from the CATALOG, using their exact titles and authors. Never mention any other book, even famous ones.
- For each book you suggest, give a short reason it fits, using the relevant part of its catalog entry.
- The user already owns these books, so never suggest them: {owned_books}
- Only state a publication year if it appears in the book's catalog entry.
- If no book in the CATALOG fits the question, say you don't know. Do not guess.

Answer format (a numbered list):
1. **<Title>** by <Author>: <why it fits>

CATALOG:
{context}"""),
    ("user", "{question}")
])


def get_rag_chain():
    # Retrieval happens before the chain (in prepare_answer), so the chain is just prompt -> LLM -> text
    return rag_prompt | rag_llm | StrOutputParser()


rag_chain = get_rag_chain()


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()


def _parse_user_book(entry: str) -> tuple[str, str]:
    """'Dune by Frank Herbert' -> ('dune', 'herbert'). Author is '' if not given."""
    title, _, author = entry.rpartition(" by ") if " by " in entry else (entry, "", "")
    last_name = _norm(author).split()[-1:] or [""]
    return _norm(title), last_name[0]


def _is_owned(book: dict[str, Any], owned: list[tuple[str, str]]) -> bool:
    """Title must match (allowing small typos and collections like 'Dune, Dune Messiah...');
    the author must match too when the user gave one, so 'The Martian' by Andy Weir
    doesn't hide 'The Martian Chronicles' by Ray Bradbury."""
    title_words = _norm(book["title"]).split()
    author_words = _norm(" ".join(book["authors"])).split()
    for owned_title, owned_last_name in owned:
        if not owned_title:
            continue
        prefix = " ".join(title_words[: len(owned_title.split())])
        if SequenceMatcher(None, owned_title, prefix).ratio() < 0.85:
            continue
        if not owned_last_name or any(
            SequenceMatcher(None, owned_last_name, word).ratio() >= 0.8 for word in author_words
        ):
            return True
    return False


def retrieve_books(question: str, top_k: int, user_books: list[str] | None = None) -> list[dict[str, Any]]:
    """Return up to top_k distinct books, best first, each with the chunks that matched."""
    owned = [_parse_user_book(entry) for entry in user_books or []]
    books: dict[str, dict[str, Any]] = {}
    for hit in search_collection(question, k=top_k * CHUNKS_PER_BOOK):
        meta = hit["metadata"]
        book = books.get(meta["work_id"])
        if book is None:  # Results arrive closest first, so the first chunk is the book's best
            book = books[meta["work_id"]] = {
                "work_id": meta["work_id"],
                "title": meta["title"],
                "authors": meta.get("authors", []),
                "first_publish_year": meta.get("first_publish_year"),
                "source_url": meta.get("source_url"),
                "distance": hit["distance"],
                "chunks": [],
            }
        book["chunks"].append(hit["text"])
    kept = [book for book in books.values() if not _is_owned(book, owned)]
    return kept[:top_k]


def _context(books: list[dict[str, Any]]) -> str:
    return "\n\n".join(
        f"[{number}]\n" + "\n\n".join(book["chunks"]) for number, book in enumerate(books, 1)
    )


def _source(book: dict[str, Any]) -> dict[str, Any]:
    """The shape the frontend and eval read: title, authors (list), source_url, score."""
    return {
        "work_id": book["work_id"],
        "title": book["title"],
        "authors": book["authors"],
        "first_publish_year": book["first_publish_year"],
        "source_url": book["source_url"],
        "score": round(1 - book["distance"], 4),  # Cosine similarity, higher is better
        "distance": round(book["distance"], 4),
    }


def prepare_answer(question: str, top_k: int, user_books: list[str] | None = None) -> dict[str, Any]:
    """Retrieve and decide whether to refuse. Shared by the normal and streaming endpoints."""
    books = retrieve_books(question, top_k, user_books)
    if not books or books[0]["distance"] > REFUSAL_DISTANCE:
        # Refusals return no sources: nothing in the catalog was close enough to cite
        return {"refused": True, "sources": [], "inputs": None}
    return {
        "refused": False,
        "sources": [_source(book) for book in books],
        "inputs": {
            "context": _context(books),
            "question": question,
            "owned_books": "; ".join(user_books) if user_books else "none",
        },
    }


def answer_question(question: str, top_k: int = 5, user_books: list[str] | None = None) -> dict[str, Any]:
    started = time.perf_counter()
    prepared = prepare_answer(question, top_k, user_books)
    answer = REFUSAL_MESSAGE if prepared["refused"] else rag_chain.invoke(prepared["inputs"])
    return {
        "answer": answer,
        "sources": prepared["sources"],
        "refused": prepared["refused"],
        "latency_ms": int((time.perf_counter() - started) * 1000),
    }


def stream_answer(question: str, top_k: int = 5, user_books: list[str] | None = None) -> Iterator[str | dict]:
    """Yield answer text pieces, then one final dict with sources, refused and latency_ms."""
    started = time.perf_counter()
    prepared = prepare_answer(question, top_k, user_books)
    if prepared["refused"]:
        yield REFUSAL_MESSAGE
    else:
        for piece in rag_chain.stream(prepared["inputs"]):
            if piece:
                yield piece
    yield {
        "sources": prepared["sources"],
        "refused": prepared["refused"],
        "latency_ms": int((time.perf_counter() - started) * 1000),
    }
