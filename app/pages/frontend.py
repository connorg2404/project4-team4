# Librarian Bot - Streamlit frontend (Person 5)
#
# Talks to the team API contract:
#   POST /ask         {"question": str, "top_k": int, "user_books": [str]  <- optional}
#                     -> {"answer", "sources": [{"title","authors","source_url","score"}],
#                         "refused": bool, "latency_ms": int}
#   POST /ask/stream  same request, server-sent events: token chunks, then a final
#                     "sources" event carrying the same fields as /ask (minus the answer)
#   GET  /health      -> {"status", "model", "index_size"}
#
# "user_books" is the list of books the user owns or has read, one string per book
# ("Title" or "Title by Author"). It is only sent when the list is not empty. It is a
#
# Run from the project root:   streamlit run frontend.py
 
import json
import os
import time
 
import requests
import streamlit as st
 
DEFAULT_API_URL = os.getenv("LIBRARIAN_API_URL", "http://localhost:8000")
REQUEST_TIMEOUT_S = 60
REFUSAL_FALLBACK = "I don't know. I couldn't find anything relevant in the catalog."
MAX_BOOKS = 100 
MAX_BOOK_CHARS = 200
RECOMMEND_PROMPT = "Based on the books on my list, what should I read next?"
 
st.set_page_config(page_title="Librarian Bot", page_icon="📚")
 
# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------
class ApiError(Exception):
    """An error with a message that is safe to show to the user."""
 
def describe_error(exc: Exception, api_url: str) -> str:
    if isinstance(exc, ApiError):
        return str(exc)
    if isinstance(exc, requests.exceptions.ConnectionError):
        return f"Can't reach the API at {api_url}. Is the backend running?"
    if isinstance(exc, requests.exceptions.Timeout):
        return "The API took too long to respond. Please try again in a moment."
    if isinstance(exc, requests.exceptions.HTTPError):
        status = exc.response.status_code if exc.response is not None else "?"
        return f"The API returned an error (HTTP {status})."
    if isinstance(exc, ValueError):
        return "The API sent back a response I couldn't read."
    return f"Something went wrong: {exc}"
 
def build_payload(question: str, top_k: int, user_books: list[str]) -> dict:
    payload = {"question": question, "top_k": top_k}
    if user_books:  # leave the field out entirely when there is nothing to send
        payload["user_books"] = user_books
    return payload
 
def check_health(api_url: str) -> dict:
    resp = requests.get(f"{api_url}/health", timeout=5)
    resp.raise_for_status()
    return resp.json()
 
def ask_once(api_url: str, payload: dict) -> dict:
    resp = requests.post(f"{api_url}/ask", json=payload, timeout=REQUEST_TIMEOUT_S)
    resp.raise_for_status()
    return resp.json()
 
def parse_sse(resp: requests.Response):
    """Yield (event_name, data_string) pairs from a server-sent-events response."""
    resp.encoding = "utf-8"
    event, data_lines = "message", []
    for line in resp.iter_lines(decode_unicode=True):
        if line is None:
            continue
        if line == "":  # blank line = end of one event
            if data_lines:
                yield event, "\n".join(data_lines)
            event, data_lines = "message", []
        elif line.startswith(":"):  # comment / keep-alive
            continue
        elif line.startswith("event:"):
            event = line[6:].strip()
        elif line.startswith("data:"):
            value = line[5:]
            data_lines.append(value[1:] if value.startswith(" ") else value)
    if data_lines:
        yield event, "\n".join(data_lines)
 
def token_from_data(data: str) -> str:
    """Tokens may arrive as raw text or as JSON like {"token": "..."}; accept both."""
    if data.startswith("{"):
        try:
            payload = json.loads(data)
            if isinstance(payload, dict):
                return str(payload.get("token", payload.get("text", "")))
        except ValueError:
            pass
    return data
 
def stream_answer(api_url: str, payload: dict, result: dict):
    """Generator of answer tokens. Fills `result` with sources/refused/latency at the end."""
    with requests.post(
        f"{api_url}/ask/stream", json=payload, stream=True, timeout=REQUEST_TIMEOUT_S
    ) as resp:
        resp.raise_for_status()
        for event, data in parse_sse(resp):
            if event == "sources":
                result.update(json.loads(data))
            elif event == "error":
                raise ApiError(f"The API reported an error: {data}")
            elif event == "done":
                break
            else:
                yield token_from_data(data)
 
# ---------------------------------------------------------------------------
# "My books" list helpers
# ---------------------------------------------------------------------------
def add_books(candidates: list[str]) -> tuple[int, int]:
    """Add cleaned, de-duplicated entries to the user's list. Returns (added, skipped)."""
    books = st.session_state.my_books
    seen = {b.lower() for b in books}
    added = skipped = 0
    for raw in candidates:
        book = " ".join(raw.split())[:MAX_BOOK_CHARS]
        if not book:
            continue
        if book.lower() in seen or len(books) >= MAX_BOOKS:
            skipped += 1
            continue
        books.append(book)
        seen.add(book.lower())
        added += 1
    return added, skipped
 
def toast_add_result(added: int, skipped: int) -> None:
    msg = f"Added {added} book{'s' if added != 1 else ''}."
    if skipped:
        msg += f" Skipped {skipped} (duplicates, or the list is full at {MAX_BOOKS})."
    st.toast(msg)
 
# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------
def render_sources(sources: list) -> None:
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})"):
        for src in sources:
            title = src.get("title") or "Untitled"
            url = src.get("source_url")
            authors = ", ".join(src.get("authors") or [])
            score = src.get("score")
            line = f"[{title}]({url})" if url else title
            if authors:
                line += f" by {authors}"
            if isinstance(score, (int, float)):
                line += f" (score {score:.2f})"
            st.markdown(f"- {line}")
 
def render_assistant(msg: dict) -> None:
    if msg.get("error"):
        st.error(msg["error"])
        return
    if msg.get("refused"):
        st.info(msg.get("content") or REFUSAL_FALLBACK)
    else:
        st.markdown(msg["content"])
    render_sources(msg.get("sources", []))
    if msg.get("latency_ms") is not None:
        st.caption(f"Answered in {msg['latency_ms'] / 1000:.1f}s")
 
def get_assistant_message(
    api_url: str, question: str, top_k: int, use_stream: bool, user_books: list[str]
) -> dict:
    """Call the API (streaming or not), draw the live output, and return the message dict."""
    started = time.perf_counter()
    payload = build_payload(question, top_k, user_books)
    msg = {"role": "assistant", "content": "", "sources": [], "refused": False}
    try:
        if use_stream:
            box = st.empty()
            box.markdown("_Searching the catalog..._")
            result: dict = {}
            with box.container():
                msg["content"] = st.write_stream(stream_answer(api_url, payload, result))
            box.empty()
            msg["sources"] = result.get("sources", [])
            msg["refused"] = bool(result.get("refused", False))
            msg["latency_ms"] = result.get("latency_ms")
        else:
            with st.spinner("Searching the catalog..."):
                data = ask_once(api_url, payload)
            msg["content"] = data.get("answer", "")
            msg["sources"] = data.get("sources", [])
            msg["refused"] = bool(data.get("refused", False))
            msg["latency_ms"] = data.get("latency_ms")
        if msg.get("latency_ms") is None:  # fall back to our own timing
            msg["latency_ms"] = int((time.perf_counter() - started) * 1000)
    except Exception as exc:  # noqa: BLE001 - every failure becomes a friendly message
        msg = {"role": "assistant", "content": "", "error": describe_error(exc, api_url)}
    render_assistant(msg)
    return msg
 
# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "my_books" not in st.session_state:
    st.session_state.my_books = []
 
st.title("📚 Librarian Bot")
st.caption(
    "Ask about recent bestsellers or get recommendations. "
    "Answers come only from our bestseller data, with sources."
)
 
with st.sidebar:
    st.header("Settings")
    api_url = st.text_input("API URL", value=DEFAULT_API_URL).rstrip("/")
    top_k = st.slider("Sources to retrieve (top_k)", min_value=1, max_value=10, value=4)
    use_stream = st.toggle("Stream answers", value=True)
 
    if st.button("Check connection"):
        try:
            health = check_health(api_url)
            st.success(f"OK. Model: {health.get('model')}, index size: {health.get('index_size')}")
        except Exception as exc:  # noqa: BLE001
            st.error(describe_error(exc, api_url))
 
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()
 
    st.divider()
    st.header("My books")
    st.caption(
        "Books you own or have read. They are sent with each question so the bot can "
        "personalize recommendations and skip books you already have."
    )
 
    with st.form("add_book", clear_on_submit=True):
        new_title = st.text_input("Title")
        new_author = st.text_input("Author (optional)")
        if st.form_submit_button("Add book") and new_title.strip():
            entry = new_title.strip()
            if new_author.strip():
                entry += f" by {new_author.strip()}"
            toast_add_result(*add_books([entry]))
 
    with st.expander("Add many at once"):
        with st.form("bulk_add", clear_on_submit=True):
            pasted = st.text_area(
                "One book per line",
                height=120,
                placeholder="Dune by Frank Herbert\nNeuromancer",
            )
            uploaded = st.file_uploader("Or upload a .txt or .csv (one book per line)", type=["txt", "csv"])
            if st.form_submit_button("Add to my list"):
                lines = pasted.splitlines()
                if uploaded is not None:
                    lines += uploaded.getvalue().decode("utf-8", errors="replace").splitlines()
                toast_add_result(*add_books(lines))
 
    books = st.session_state.my_books
    if books:
        st.caption(f"{len(books)} of {MAX_BOOKS} books")
        for i, book in enumerate(books):
            name_col, remove_col = st.columns([5, 1])
            name_col.write(book)
            if remove_col.button("✕", key=f"remove_{i}", help="Remove this book"):
                books.pop(i)
                st.rerun()
        if st.button("Recommend me something"):
            st.session_state.pending_question = RECOMMEND_PROMPT
        if st.button("Clear my list"):
            st.session_state.my_books = []
            st.rerun()
    else:
        st.caption("No books added yet.")
 
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.markdown(message["content"])
        else:
            render_assistant(message)
 
question = st.chat_input("Ask about a book, an author, or a topic...") or st.session_state.pop(
    "pending_question", None
)
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        reply = get_assistant_message(api_url, question, top_k, use_stream, list(st.session_state.my_books))
    st.session_state.messages.append(reply)
