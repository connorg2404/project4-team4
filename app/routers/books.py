# FastAPI Routers share a use case with Spring Controllers
# They take in HTTP requests, and return HTTP responses
from fastapi import APIRouter
# from langchain_community.document_loaders import TextLoader, CSVLoader
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel

from app.models.book_model import BookModel
from app.services.langchain_service import get_general_chain, get_memory_chain, get_rag_chain
from .books_db import book_db

# Every router gets set up like this:
router = APIRouter(
    prefix="/books", # HTTP requests to /books will route here
    tags=["books"] # This is for the Swagger UI docs
)

class ChatInputModel(BaseModel):
    input:str

chain = get_general_chain()
rag_chain = get_rag_chain()
@router.post("/chat")
async def general_chat(chat:ChatInputModel):
    return chain.invoke({"input":chat.input})




# How can we get the LLM to answer questions based on the monster data?
# This is our first look at RAG which you'll need for your project
#book_db will be replaced with the vector_db others are working on as the data.
@router.post("/rag_chat")
async def rag_chat(chat:ChatInputModel):

    # This time, lets add the monster data plus some extra prompting
    # Remember this is called a PROMPT TEMPLATE
    # A prompt that takes in {variables} for dynamic behavior
    return rag_chain.invoke({
        "input": f"""
     You are a librarian for a small catalog. The CATALOG in the user's message is the 
     ONLY set of books that exist. Only mention books from the CATALOG, using their exact titles. 
     Never mention any other book, even famous ones. Give a short explanation for why you chose these books, 
     providing the relevant portion of the description from the database.
     
    
        
                    
            The CATALOG: {book_db}
            
            The User's query: {chat.input}
        """
    })


#@router.post("/book_suggest_genre", status_code=200)
#async def rag_with_genre(chat:ChatInputModel)

#@router.post("/book_suggest_author", status_code=200)
#async def rag_with_author(chat:ChatInputModel)

#@router.post("/book_suggest", status_code=200)
#async def rag_with_keyword(chat:ChatInputModel)