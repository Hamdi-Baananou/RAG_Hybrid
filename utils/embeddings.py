import time
from langchain_community.embeddings import HuggingFaceEmbeddings
from utils.logging_config import logger

def get_embeddings():
    """Initialize and return embedding model"""
    logger.info("Initializing embedding model")
    start_time = time.time()
    
    try:
        embedding_function = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        logger.info(f"Embedding model loaded in {time.time() - start_time:.2f} seconds")
        return embedding_function
    except Exception as e:
        logger.error(f"Failed to initialize embedding model: {str(e)}", exc_info=True)
        raise 