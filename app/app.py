from fastapi import FastAPI

from app.routers import books

# IMPORTANT: From the main project folder, run the app using the following:
# python -m uvicorn app.app:app --reload

# Set up FastAPI. We'll use this "app" variable to do FastAPI stuff below
app = FastAPI()

# TODO: Global exception handling with @app.exception_handler

# import/activate the routers here
app.include_router(books.router)

# Generic sample endpoint (GET request with a greeting)
@app.get("/")
async def sample_endpoint():
    return {"greeting":"Hello from FastAPI! This is our project backend library"}