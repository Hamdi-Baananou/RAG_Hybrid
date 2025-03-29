import time
from langchain_community.embeddings import HuggingFaceEmbeddings
from utils.logging_config import logger
from langchain_openai import OpenAIEmbeddings
from utils.graph_db import get_api_credentials

def get_embeddings():
    """
    Create an embedding function that outputs vectors with 1536 dimensions
    to match your existing Neo4j vector index
    
    Use an embedding model that can produce 1536-dimensional vectors
    like OpenAI's ada-002 or newer models
    """
    # Check if you're using langchain HuggingFaceEmbeddings, which might be generating 384-dim vectors
    # Replace with a model that produces 1536-dim vectors to match your index
    logger.info("Initializing embedding model")
    start_time = time.time()
    
    try:
        # Get credentials using existing helper
        credentials = get_api_credentials()
        
        # Use OpenAI embeddings which are 1536 dimensions
        embeddings = OpenAIEmbeddings(
            model="text-embedding-ada-002",  # This model produces 1536-dim vectors
            openai_api_key=credentials["deepseek_api_key"],
            openai_api_base="https://api.deepseek.com/v1"
        )
        logger.info(f"Embedding model loaded in {time.time() - start_time:.2f} seconds")
        return embeddings
    except Exception as e:
        logger.error(f"Failed to initialize embedding model: {str(e)}", exc_info=True)
        raise 