# FastAPI Routers share a use case with Spring Controllers
# They take in HTTP requests, and return HTTP responses
from fastapi import APIRouter, HTTPException
# from langchain_community.document_loaders import TextLoader, CSVLoader
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, model_validator

from app.models.book_model import BookModel
from app.services.ingest_service import add_book
from app.services.langchain_service import answer_question, get_general_chain

# Every router gets set up like this:
router = APIRouter(
    prefix="/books", # HTTP requests to /books will route here
    tags=["books"] # This is for the Swagger UI docs
)

class ChatInputModel(BaseModel):
    input:str

chain = get_general_chain()
@router.post("/chat")
async def general_chat(chat:ChatInputModel):
    return chain.invoke({"input":chat.input})




# RAG chat: retrieves books from the vector DB, then the LLM answers using only those books.
# Same logic as POST /ask (see vectordb_routes.py), with this router's simpler input.
# Plain `def` because retrieval and Bedrock calls block (FastAPI runs it in a thread pool).
@router.post("/rag_chat")
def rag_chat(chat:ChatInputModel):
    return answer_question(chat.input, top_k=5)


class NewBook(BaseModel):
    """A book to add to the searchable catalog."""
    title: str = Field(min_length=1, max_length=300, examples=["The Lantern Keeper"])
    authors: list[str] = Field(min_length=1, max_length=10, examples=[["Mara Quill"]])
    description: str = Field(min_length=20, max_length=20000)
    subjects: list[str] = Field(default_factory=list, max_length=50, examples=[["fantasy", "lighthouses"]])
    first_publish_year: int | None = Field(None, ge=1, le=2100)
    author_bio: str | None = Field(None, max_length=5000)
    author_bio_author: str | None = Field(None, description="Which author the bio is about. Defaults to the first author.")

    @model_validator(mode="after")
    def _check(self):
        self.authors = [name.strip() for name in self.authors]
        if not all(self.authors) or not self.title.strip() or not self.description.strip():
            raise ValueError("title, authors and description must not be blank.")
        if self.author_bio and not self.author_bio_author:
            self.author_bio_author = self.authors[0]
        if self.author_bio_author and self.author_bio_author not in self.authors:
            raise ValueError("author_bio_author must be one of the authors.")
        return self


# Add a book to the catalog. Plain `def`: embedding blocks, so FastAPI runs it in a thread pool.
@router.post("/ingest", status_code=201)
def ingest_book(book: NewBook):
    try:
        return add_book(book.model_dump())
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error))


#@router.post("/book_suggest_genre", status_code=200)
#async def rag_with_genre(chat:ChatInputModel)

#@router.post("/book_suggest_author", status_code=200)
#async def rag_with_author(chat:ChatInputModel)

#@router.post("/book_suggest", status_code=200)
#async def rag_with_keyword(chat:ChatInputModel)