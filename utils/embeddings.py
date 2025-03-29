import time
from langchain_community.embeddings import HuggingFaceEmbeddings
from utils.logging_config import logger

def get_embeddings():
    """
    Create an embedding function that outputs vectors with 1536 dimensions
    to match your existing Neo4j vector index
    
    Use an embedding model that can produce 1536-dimensional vectors
    like OpenAI's ada-002 or newer models
    """
    logger.info("Initializing embedding model")
    start_time = time.time()
    
    try:
        # Use a model that generates 384-dimensional embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        logger.info(f"Embedding model loaded in {time.time() - start_time:.2f} seconds")
        return embeddings
    except Exception as e:
        logger.error(f"Failed to initialize embedding model: {str(e)}", exc_info=True)
        raise 