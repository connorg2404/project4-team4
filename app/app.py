import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.concurrency import run_in_threadpool

from app.routers import books, vectordb_routes
from app.services.vectordb_service import warm_up

# IMPORTANT: From the main project folder, run the app using the following:
# python -m uvicorn app.app:app --reload


# Runs once when the server starts: load the embedding model now, so the
# first question doesn't wait for it (and eval latency numbers stay honest)
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        await run_in_threadpool(warm_up)
    except Exception:  # Don't stop the server; /ask will report the real error
        logging.getLogger(__name__).exception("Embedding model warm-up failed")
    yield


# Set up FastAPI. We'll use this "app" variable to do FastAPI stuff below
app = FastAPI(lifespan=lifespan)

# TODO: Global exception handling with @app.exception_handler

# import/activate the routers here
app.include_router(books.router)
app.include_router(vectordb_routes.router)  # /health, /ask, /ask/stream

# Generic sample endpoint (GET request with a greeting)
@app.get("/")
async def sample_endpoint():
    return {"greeting":"Hello from FastAPI! This is our project backend library"}
