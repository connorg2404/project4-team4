
from langchain.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever


# # Similarity search endpoint for MONSTER FIGHTING TOOLS
# # (NOT RAG yet)
# @router.get("/search_monster_tools")
# async def search_tools(query:str, k:int=5):
#
#
#     # Call the Service method to search the collection
#     results = search_collection(
#         collection_name="monster_fighting_tools",
#         query=query,
#         # We COULD set the k value here, or just let default to 5
#         k=k
#     )
#
#     return results



# RAG endpoint for MONSTER FIGHTING TOOLS
@router.get("/rag_books_suggestions")
async def tools_rag(query:str):

    # First, get the search results from the collection
    results = search_collection(
        collection_name="books",
        query=query
        # We COULD set the k value here, or just let default to 5
    )

    # Here's the RAG part -
    # Augmenting the Generated response with the data we Retrieved
    return chain.invoke({
        "input": f"""
            Respond to the user's query by suggesting books
            
            ONLY use info from the search results, 
            but you can be creative about how to use them.
            
            Don't fall back to outside info, and you can say you don't know.
            
            The User's Query: {query}
            
            The Search Results: {results}
            
        """
    })


