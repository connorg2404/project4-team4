# FastAPI Routers share a use case with Spring Controllers
# They take in HTTP requests, and return HTTP responses
from fastapi import APIRouter
from langchain_community.document_loaders import TextLoader, CSVLoader
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import SecretStr, BaseModel

from app.models.monster_model import MonsterModel
from app.services.langchain_service import get_general_chain, get_memory_chain
from app.services.langgraph_service import agentic_graph

# Every router gets set up like this:
router = APIRouter(
    prefix="/books", # HTTP requests to /books will route here
    tags=["books"] # This is for the Swagger UI docs
)

@router.post("/book_suggest_genre", status_code=200)
async def rag_with_genre(chat:ChatInputModel):

@router.post("/book_suggest_author", status_code=200)
async def rag_with_author(chat:ChatInputModel):

@router.post("/book_suggest", status_code=200)
async def rag_with_keyword(chat:ChatInputModel):