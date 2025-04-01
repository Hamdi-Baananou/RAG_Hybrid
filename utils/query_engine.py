# /mount/src/rag_hybrid/utils/query_engine.py # Assuming this path based on logging_config import

import re
import time
import json
import random  # Adding missing import for random.uniform
from typing import List, Dict, Any, Optional, Tuple
from langchain_community.graphs import Neo4jGraph
# from langchain.vectorstores.neo4j_vector import Neo4jVector # Original outdated import
from langchain_community.vectorstores import Neo4jVector # Corrected import
from openai import OpenAI  # Add this import
from utils.logging_config import logger # Assuming logger is configured correctly
import logging
from neo4j import GraphDatabase

# --- Option 1: Keep Simple Entity Extractor (with improvements) ---
# def extract_entities_from_question_simple(question: str, min_len: int = 3) -> List[str]:
#     """Simple keyword extraction (improved slightly)"""
#     logger.debug(f"Extracting entities simply from: {question}")
#     stop_words = {'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
#                   'with', 'by', 'about', 'like', 'through', 'over', 'before', 'after',
#                   'since', 'of', 'from', 'is', 'was', 'were', 'be', 'are', 'what', 'who',
#                   'when', 'where', 'why', 'how', 'which', 'does', 'do', 'did'}

#     # Normalize: lower case and remove punctuation
#     normalized_question = re.sub(r'[^\w\s]', '', question.lower())

#     # Tokenize and filter stop words and short words
#     words = [word for word in normalized_question.split() if word not in stop_words and len(word) >= min_len]

#     # Rudimentary phrase extraction (consider adjacent non-stop words) - Less reliable
#     # For better results, consider NLP libraries or LLM extraction below

#     entities = list(set(words)) # Use unique words for now
#     logger.debug(f"Simple extracted entities: {entities}")
#     return entities

# --- Option 2: Use LLM for Entity Extraction (Recommended for better accuracy) ---
def extract_entities_with_llm(
    question: str,
    api_key: str,
    model: str = "deepseek-chat",  # Changed from Fireworks model
    max_retries: int = 3,
    initial_backoff: float = 2.0
    ) -> List[str]:
    """Extracts named entities and key concepts from the question using DeepSeek LLM."""
    logger.debug(f"Extracting entities via DeepSeek LLM from: {question}")
    
    # Initialize OpenAI client with DeepSeek base URL
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com", timeout=30.0)
    
    # Small proactive delay before making API call to avoid burst limits
    time.sleep(random.uniform(0.1, 0.3))
    
    # Simple prompt for entity extraction
    prompt = f"""
    Extract the key named entities (like people, organizations, locations, products, technologies)
    and essential concepts or topics from the following question.
    Return ONLY a comma-separated list of these terms. Do not include explanations or introductory text.

    Question: {question}

    Comma-separated list:
    """

    current_retry = 0
    backoff_time = initial_backoff
    while current_retry < max_retries:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that extracts entities from text."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=100,
                temperature=0.0,
            )

            if response.choices and len(response.choices) > 0:
                raw_entities = response.choices[0].message.content.strip()
                # Clean up the list: split by comma, strip whitespace, remove empty strings
                entities = [e.strip() for e in raw_entities.split(',') if e.strip()]
                logger.debug(f"LLM extracted entities: {entities}")
                return entities
            else:
                logger.warning("LLM entity extraction returned no choices.")
                return [] # Return empty list on failure

        except Exception as e:
            current_retry += 1
            if current_retry >= max_retries:
                 logger.error(f"LLM entity extraction failed after {max_retries} retries: {e}")
                 return [] # Return empty list on final failure
            else:
                 # Add more jitter to avoid synchronized retries
                 wait_time = backoff_time + random.uniform(0, backoff_time)
                 logger.warning(f"LLM entity extraction attempt {current_retry} failed ({e}). Retrying in {wait_time:.2f}s...")
                 time.sleep(wait_time)
                 backoff_time = min(backoff_time * 2, 30.0) # Exponential backoff capped at 30s

    return [] # Should not be reached if loop logic is correct


def get_query_context(
    question: str,
    vector_store: Any,
    graph: Any,
    api_key: str,
    k_vector: int = 3,
    k_graph: int = 5,
    use_llm_extraction: bool = True
) -> Dict[str, Any]:
    """Get enriched context by combining vector search results with graph traversal

    Args:
        question: User question or prompt
        vector_store: Vector store for semantic search
        graph: Neo4j graph connection
        api_key: API key for LLM
        k_vector: Number of vector results to retrieve
        k_graph: Number of graph results to retrieve
        use_llm_extraction: Whether to use LLM for extraction

    Returns:
        Combined context from vector and graph sources
    """
    # 1. Get vector-based context
    try:
        vector_results = vector_store.similarity_search(question, k=k_vector)
        vector_context = "\n\n".join([doc.page_content for doc in vector_results])
    except Exception as e:
        logging.error(f"Error in vector retrieval: {str(e)}")
        vector_context = "Vector retrieval failed."

    # 2. Get graph-based context with specialized queries
    graph_context = ""
    try:
        # Determine query type based on question content
        query_type = determine_query_type(question)
        
        # Get graph results using specialized queries
        graph_results = run_specialized_query(graph, query_type, question, k_graph)
        
        # Format graph results
        if graph_results:
            graph_context = format_graph_results(graph_results, query_type)
        else:
            graph_context = "No relevant graph connections found."
    except Exception as e:
        logging.error(f"Error in graph retrieval: {str(e)}")
        graph_context = "Graph retrieval failed."

    # Return combined contexts
    return {
        "vector_context": vector_context,
        "graph_context": graph_context,
        "query_type": query_type
    }

def determine_query_type(question: str) -> str:
    """Determine query type based on question content"""
    question_lower = question.lower()
    
    # Material properties
    if any(term in question_lower for term in ["material", "polymer", "plastic", "pa", "pbt"]):
        return "material"
    
    # Physical dimensions
    elif any(term in question_lower for term in ["height", "length", "width", "dimension", "mm", "size"]):
        return "dimension"
    
    # Gender/connectors
    elif any(term in question_lower for term in ["gender", "male", "female", "connector"]):
        return "gender"
    
    # Cavities/rows
    elif any(term in question_lower for term in ["cavity", "cavities", "row", "rows"]):
        return "cavity"
    
    # Sealing
    elif any(term in question_lower for term in ["seal", "sealing", "waterproof", "ip"]):
        return "sealing"
    
    # Color
    elif any(term in question_lower for term in ["color", "colour", "black", "white"]):
        return "color"
    
    # Temperature
    elif any(term in question_lower for term in ["temperature", "thermal", "heat", "degree"]):
        return "temperature"
    
    # Default
    return "general"

def run_specialized_query(graph: Any, query_type: str, question: str, limit: int) -> List[Dict[str, Any]]:
    """Run specialized graph query based on query type"""
    
    # Material-specific query
    if query_type == "material":
        query = """
        MATCH (doc:Document)-[:CONTAINS]->(chunk:Chunk)
        WHERE chunk.text CONTAINS 'material' OR chunk.text CONTAINS 'polymer' OR chunk.text CONTAINS 'plastic'
        WITH doc, chunk
        OPTIONAL MATCH (chunk)-[:HAS_ENTITY]->(material:Entity)
        WHERE material.type IN ['MATERIAL', 'CHEMICAL', 'SUBSTANCE'] OR 
              material.text CONTAINS 'PA' OR 
              material.text CONTAINS 'PBT' OR
              material.text CONTAINS 'Nylon'
        RETURN doc.title as document, chunk.page as page, chunk.text as context, 
               material.text as entity_text, material.type as entity_type,
               exists((chunk)-[:HAS_ENTITY]->(material)) as has_material
        ORDER BY has_material DESC, material.text
        LIMIT $limit
        """
    
    # Dimension-specific query
    elif query_type == "dimension":
        query = """
        MATCH (doc:Document)-[:CONTAINS]->(chunk:Chunk)
        WHERE chunk.text CONTAINS 'mm' OR chunk.text CONTAINS 'dimension' OR 
              chunk.text CONTAINS 'height' OR chunk.text CONTAINS 'width' OR chunk.text CONTAINS 'length'
        WITH doc, chunk
        OPTIONAL MATCH (chunk)-[:HAS_ENTITY]->(dimension:Entity)
        WHERE dimension.type = 'QUANTITY' OR dimension.text CONTAINS 'mm'
        RETURN doc.title as document, chunk.page as page, chunk.text as context,
               dimension.text as entity_text, dimension.type as entity_type,
               exists((chunk)-[:HAS_ENTITY]->(dimension)) as has_dimension
        ORDER BY has_dimension DESC
        LIMIT $limit
        """
    
    # Gender-specific query
    elif query_type == "gender":
        query = """
        MATCH (doc:Document)-[:CONTAINS]->(chunk:Chunk)
        WHERE chunk.text CONTAINS 'male' OR chunk.text CONTAINS 'female' OR chunk.text CONTAINS 'gender'
        WITH doc, chunk
        OPTIONAL MATCH (chunk)-[:HAS_ENTITY]->(gender:Entity)
        WHERE gender.text IN ['male', 'female'] OR gender.text CONTAINS 'connector'
        RETURN doc.title as document, chunk.page as page, chunk.text as context,
               gender.text as entity_text, gender.type as entity_type,
               exists((chunk)-[:HAS_ENTITY]->(gender)) as has_gender_info
        ORDER BY has_gender_info DESC
            LIMIT $limit
            """
            
    # Cavity-specific query
    elif query_type == "cavity":
        query = """
        MATCH (doc:Document)-[:CONTAINS]->(chunk:Chunk)
        WHERE chunk.text CONTAINS 'cavity' OR chunk.text CONTAINS 'cavities' OR 
              chunk.text CONTAINS 'row' OR chunk.text CONTAINS 'rows'
        WITH doc, chunk
        OPTIONAL MATCH (chunk)-[:HAS_ENTITY]->(num:Entity)
        WHERE num.type = 'CARDINAL' OR num.type = 'QUANTITY'
        RETURN doc.title as document, chunk.page as page, chunk.text as context,
               num.text as entity_text, num.type as entity_type,
               exists((chunk)-[:HAS_ENTITY]->(num)) as has_number
        ORDER BY has_number DESC
        LIMIT $limit
        """
    
    # Sealing-specific query
    elif query_type == "sealing":
        query = """
        MATCH (doc:Document)-[:CONTAINS]->(chunk:Chunk)
        WHERE chunk.text CONTAINS 'seal' OR chunk.text CONTAINS 'waterproof' OR 
              chunk.text CONTAINS 'IP' OR chunk.text CONTAINS 'protection'
        WITH doc, chunk
        OPTIONAL MATCH (chunk)-[:HAS_ENTITY]->(seal:Entity)
        WHERE seal.text CONTAINS 'seal' OR seal.text CONTAINS 'IP'
        RETURN doc.title as document, chunk.page as page, chunk.text as context,
               seal.text as entity_text, seal.type as entity_type,
               exists((chunk)-[:HAS_ENTITY]->(seal)) as has_sealing
        ORDER BY has_sealing DESC
        LIMIT $limit
        """
    
    # Color-specific query
    elif query_type == "color":
        query = """
        MATCH (doc:Document)-[:CONTAINS]->(chunk:Chunk)
        WHERE chunk.text CONTAINS 'color' OR chunk.text CONTAINS 'colour' OR 
              chunk.text CONTAINS 'black' OR chunk.text CONTAINS 'white'
        WITH doc, chunk
        OPTIONAL MATCH (chunk)-[:HAS_ENTITY]->(color:Entity)
        WHERE color.type = 'COLOR' OR color.text IN ['black', 'white', 'red', 'blue', 'green', 'yellow']
        RETURN doc.title as document, chunk.page as page, chunk.text as context,
               color.text as entity_text, color.type as entity_type,
               exists((chunk)-[:HAS_ENTITY]->(color)) as has_color
        ORDER BY has_color DESC
        LIMIT $limit
        """
    
    # Temperature-specific query
    elif query_type == "temperature":
        query = """
        MATCH (doc:Document)-[:CONTAINS]->(chunk:Chunk)
        WHERE chunk.text CONTAINS 'temperature' OR chunk.text CONTAINS '°C' OR 
              chunk.text CONTAINS 'thermal' OR chunk.text CONTAINS 'heat'
        WITH doc, chunk
        OPTIONAL MATCH (chunk)-[:HAS_ENTITY]->(temp:Entity)
        WHERE temp.type = 'QUANTITY' OR temp.text CONTAINS '°C' OR temp.text CONTAINS 'temperature'
        RETURN doc.title as document, chunk.page as page, chunk.text as context,
               temp.text as entity_text, temp.type as entity_type,
               exists((chunk)-[:HAS_ENTITY]->(temp)) as has_temp
        ORDER BY has_temp DESC
        LIMIT $limit
        """
    
    # General query for other types
    else:
        query = """
        MATCH (doc:Document)-[:CONTAINS]->(chunk:Chunk)
        WHERE chunk.text CONTAINS $keyword
        WITH doc, chunk
        OPTIONAL MATCH (chunk)-[:HAS_ENTITY]->(entity:Entity)
        RETURN doc.title as document, chunk.page as page, chunk.text as context,
               entity.text as entity_text, entity.type as entity_type,
               exists((chunk)-[:HAS_ENTITY]->(entity)) as has_entity
        ORDER BY has_entity DESC
            LIMIT $limit
            """
            
    # Extract keywords from question for general search
    keywords = extract_keywords(question)
    keyword = keywords[0] if keywords else ""
    
    # Run the query with parameters
    return graph.run(query, keyword=keyword, limit=limit).data()

def extract_keywords(text: str) -> List[str]:
    """Extract important keywords from text"""
    # Simple keyword extraction
    common_words = {'the', 'and', 'is', 'of', 'what', 'how', 'can', 'does', 'do', 'a', 'an', 'in', 'for', 'to', 'from'}
    words = [word.lower() for word in text.split() if word.lower() not in common_words and len(word) > 2]
    return words[:3]  # Return top 3 keywords

def format_graph_results(results: List[Dict[str, Any]], query_type: str) -> str:
    """Format graph results into readable context"""
    if not results:
        return "No relevant graph connections found."
    
    formatted_context = "GRAPH DATABASE RESULTS:\n\n"
    
    for result in results:
        doc = result.get('document', 'Unknown document')
        page = result.get('page', 'Unknown page')
        context = result.get('context', 'No context available')
        entity_text = result.get('entity_text', None)
        entity_type = result.get('entity_type', None)
        
        formatted_context += f"Document: {doc} (Page: {page})\n"
        formatted_context += f"Context: {context}\n"
        
        if entity_text and entity_type:
            formatted_context += f"Entity: {entity_text} (Type: {entity_type})\n"
        
        formatted_context += "\n" + "-"*50 + "\n\n"
    
    return formatted_context

def answer_question(
    question: str,
    api_key: str,
    contexts: Dict[str, Any],
    model: str = "deepseek-chat", # Added model parameter for flexibility
    max_retries: int = 3,         # Added retry logic
    initial_backoff: float = 2.0  # Added retry logic
    # Removed neo4j_uri, neo4j_username, neo4j_password as they are not used here
) -> Dict[str, Any]:
    """Generate an answer using the given contexts with DeepSeek LLM.

    Args:
        question: User question
        api_key: API key for DeepSeek LLM
        contexts: Combined contexts from vector and graph sources
        model: The DeepSeek model to use (e.g., "deepseek-chat").
        max_retries: Maximum number of retries for the API call.
        initial_backoff: Initial backoff time for retries.

    Returns:
        Dictionary with answer and metadata or error message.
    """
    start_time = time.time() # Start timing
    try:
        # Initialize OpenAI client with DeepSeek base URL
        client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com", timeout=60.0) # Increased timeout

        # Create an enhanced prompt that explicitly asks the model to use both vector and graph data
        vector_context = contexts.get("vector_context", "No vector context available.")
        graph_context = contexts.get("graph_context", "No graph context available.")
        query_type = contexts.get("query_type", "general")

        # Build a specialized system prompt based on the query type
        system_prompt = get_specialized_system_prompt(query_type)

        user_prompt = f"""
Please analyze the following information and answer this question:

QUESTION: {question}

I've retrieved information from two sources to help you:

1. VECTOR SEARCH RESULTS (text similarity):
{vector_context}

2. KNOWLEDGE GRAPH TRAVERSAL RESULTS (entity relationships):
{graph_context}

When answering, please:
1. Carefully synthesize information from BOTH the vector search results and the knowledge graph details.
2. Explicitly mention key entities, relationships, or properties found specifically in the KNOWLEDGE GRAPH RESULTS when relevant to the question.
3. If possible, cite the source document and page number for your claims based on the provided context.
4. Structure your response clearly. Start with a direct answer, then provide the supporting details and reasoning based *only* on the provided contexts.
5. If the contexts do not contain enough information to answer the question fully, state that clearly. Do not invent information.
"""
        # Retry logic for API call
        current_retry = 0
        backoff_time = initial_backoff
        while current_retry < max_retries:
            try:
                # Small delay before API call
                time.sleep(random.uniform(0.1, 0.3))

                # Generate response with the enhanced prompts using DeepSeek
                response = client.chat.completions.create(
                    model=model, # Use the specified DeepSeek model
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.0, # Lower temperature for more deterministic answers
                    max_tokens=1500 # Adjusted token limit if needed
                )

                if response.choices and len(response.choices) > 0:
                    answer = response.choices[0].message.content.strip()
                    end_time = time.time() # End timing
                    processing_time = end_time - start_time
                    logger.info(f"Answer generated successfully in {processing_time:.2f}s")

                    # Return the result
                    return {
                        "answer": answer,
                        "contexts": contexts, # Return original contexts for reference
                        "processing_time": processing_time
                    }
                else:
                    logger.warning("DeepSeek LLM answer generation returned no choices.")
                    # Optional: Retry if no choices are returned? Depends on API behavior.
                    # For now, treat as failure for this attempt.
                    raise ValueError("LLM response contained no choices.")


            except Exception as e:
                current_retry += 1
                if current_retry >= max_retries:
                    logger.error(f"DeepSeek LLM answer generation failed after {max_retries} retries: {e}")
                    end_time = time.time()
                    return {
                        "error": f"Error generating answer after multiple retries: {str(e)}",
                        "processing_time": end_time - start_time
                        }
                else:
                    wait_time = backoff_time + random.uniform(0, backoff_time / 2) # Add jitter
                    logger.warning(f"DeepSeek LLM answer generation attempt {current_retry} failed ({e}). Retrying in {wait_time:.2f}s...")
                    time.sleep(wait_time)
                    backoff_time = min(backoff_time * 2, 60.0) # Exponential backoff capped at 60s

        # This part should ideally not be reached if the loop logic is correct
        logger.error("Answer generation failed after exhausting retries (logic error).")
        end_time = time.time()
        return {
            "error": "Answer generation failed after exhausting retries.",
            "processing_time": end_time - start_time
            }


    except Exception as e:
        end_time = time.time()
        logger.error(f"Critical error during answer generation setup: {str(e)}", exc_info=True)
        return {
            "error": f"Critical error during answer generation setup: {str(e)}",
            "processing_time": end_time - start_time
            }

def get_specialized_system_prompt(query_type: str) -> str:
    """Get specialized system prompt based on query type"""
    
    base_prompt = "You are a technical documentation analysis expert specialized in automotive connectors and components."
    
    if query_type == "material":
        return base_prompt + """
Focus on material properties and classifications. When analyzing materials:
1. Identify the primary material composition (PA, PBT, etc.)
2. Note any additives or special properties (glass-filled, flame retardant)
3. Correlate the material with its standard abbreviation (e.g., Nylon = PA)
4. Distinguish between housing materials and other component materials
5. Present the evidence from both text matches and graph relationships
"""
    
    elif query_type == "dimension":
        return base_prompt + """
Focus on physical dimensions and measurements. When analyzing dimensions:
1. Extract precise measurements in millimeters
2. Distinguish between height, width, and length values
3. Note which component the dimensions apply to (housing, terminal, etc.)
4. Identify any tolerances or ranges provided
5. Present the evidence from both text matches and graph relationships
"""
    
    elif query_type == "gender":
        return base_prompt + """
Focus on connector gender and mating properties. When analyzing gender:
1. Explicitly state whether the connector is male or female
2. Note any keying or polarization features related to gender
3. Identify mating connector information if available
4. Link the gender to the specific component (housing, terminal)
5. Present the evidence from both text matches and graph relationships
"""
    
    # Add more specialized prompts for other query types
    
    else:
        return base_prompt + """
Analyze the information provided from both vector search results and knowledge graph traversal.
Pay particular attention to relationships between entities in the graph data.
Present your reasoning step-by-step, citing specific documents and page numbers.
"""

# Example Usage (Illustrative - adapt to your main script)
# if __name__ == '__main__':
#     # --- Configuration ---
#     NEO4J_URI = "bolt://localhost:7687"
#     NEO4J_USERNAME = "neo4j"
#     NEO4J_PASSWORD = "your_password"
#     DEEPSEEK_API_KEY = "your_deepseek_api_key" # <<< Use DeepSeek key
#     from langchain_community.embeddings import OllamaEmbeddings # Or your chosen embedding model
#     import os # Make sure os is imported if using environment variables

#     # --- Initialization ---
#     try:
#         # Get DeepSeek API Key (Example using environment variable)
#         api_key = os.getenv("DEEPSEEK_API_KEY", DEEPSEEK_API_KEY) # Fallback to variable if env var not set
#         if not api_key or "your_" in api_key:
#              raise ValueError("DEEPSEEK_API_KEY not configured properly.")

#         graph_db = Neo4jGraph(url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD)
#         # Configure embedding model as needed
#         # embedding_model = OllamaEmbeddings(model="nomic-embed-text")
#         # Using placeholder for embeddings if not directly needed in this example run
#         # In a real scenario, initialize your vector store properly:
#         # vector_idx = Neo4jVector(...)
#         vector_idx_placeholder = "Initialize your Neo4jVector instance here" # Placeholder

#         logger.info("Neo4j connection initialized.") # Simplified message
#     except Exception as init_error:
#         logger.critical(f"Initialization failed: {init_error}", exc_info=True)
#         exit(1)

#     # --- Query ---
#     test_question = "What is the primary material used for the connector housing described in document X page Y?" # More specific example

#     # Ensure vector_store is correctly initialized before passing
#     # For demonstration, we pass the placeholder. Replace with your actual vector_idx
#     retrieved_contexts = get_query_context(
#         question=test_question,
#         vector_store=vector_idx_placeholder, # Replace with actual vector_idx
#         graph=graph_db.query, # Pass the query execution method
#         api_key=api_key,      # Pass the DeepSeek API key
#         k_vector=3,
#         k_graph=5,
#         use_llm_extraction=True
#     )

#     print("\n--- Retrieved Contexts ---")
#     print(json.dumps(retrieved_contexts, indent=2)) # Print contexts for inspection


#     if retrieved_contexts and "error" not in retrieved_contexts:
#         final_answer_data = answer_question(
#             question=test_question,
#             api_key=api_key, # Pass DeepSeek key
#             contexts=retrieved_contexts
#             # No need to pass Neo4j credentials here anymore
#         )
#         print("\n--- Final Answer ---")
#         if "answer" in final_answer_data:
#             print(f"Answer:\n{final_answer_data['answer']}")
#         elif "error" in final_answer_data:
#             print(f"Error: {final_answer_data['error']}")
#         else:
#             print("No answer or error generated.")

#         print(f"\nProcessing Time: {final_answer_data.get('processing_time', 'N/A'):.2f}s")
#         # Example of accessing contexts if needed (they are returned in final_answer_data)
#         # print("\nContexts Used:")
#         # print(json.dumps(final_answer_data.get('contexts',{}), indent=2))

#     else:
#         print("\nContext retrieval failed or returned empty, skipping answer generation.")
#         if "error" in retrieved_contexts:
#             print(f"Retrieval Error: {retrieved_contexts['error']}")