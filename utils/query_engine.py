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
    vector_store: Neo4jVector,
    graph: Neo4jGraph,
    api_key: str, # Needed if using LLM entity extraction
    k_vector: int = 3,   # Number of results from vector search
    k_graph: int = 5,    # Max results from graph search per entity type (direct/related)
    use_llm_extraction: bool = True # Flag to choose extraction method
) -> List[Dict[str, Any]]:
    """Get context for query using vector similarity and batched graph relationships."""
    logger.info(f"Getting context for question: {question}")
    contexts_dict: Dict[str, Dict[str, Any]] = {} # Use dict for easy deduplication by chunk_id
    start_time = time.time()

    try:
        # Step 1: Extract potential entities from the question
        logger.debug("Identifying key entities in the question...")
        if use_llm_extraction:
            entities_in_question = extract_entities_with_llm(question, api_key)
        else:
            # entities_in_question = extract_entities_from_question_simple(question) # Use the simple one if preferred
            # Fallback or default simple extraction if LLM flag is false or LLM fails
            logger.warning("LLM extraction not used or failed, falling back to simple extraction.")
            # A simple split might be better than the previous regex approach
            entities_in_question = [word for word in re.findall(r'\b\w{3,}\b', question.lower()) if word not in ['the', 'a', 'an', 'is', 'of', 'for']] # Basic split
            logger.debug(f"Simple extracted entities: {entities_in_question}")


        if not entities_in_question:
             logger.warning("No entities extracted from the question. Graph search will be skipped.")

        # Step 2: Vector Similarity Search
        logger.debug(f"Performing vector similarity search (k={k_vector})...")
        try:
            vector_results = vector_store.similarity_search_with_score(question, k=k_vector)
            logger.debug(f"Vector search found {len(vector_results)} relevant chunks.")

            for idx, (doc, score) in enumerate(vector_results):
                chunk_id = doc.metadata.get('chunk_id', f"vector_unknown_{idx}") # Ensure some ID
                if chunk_id not in contexts_dict: # Add if not already present
                    contexts_dict[chunk_id] = {
                        "source": "vector_similarity",
                        "rank": idx,
                        "score": score, # Include similarity score
                        "chunk_id": chunk_id,
                        "content": doc.page_content,
                        "page_num": doc.metadata.get('page_num'), # Get page number if available
                        "source_document": doc.metadata.get('source_document', 'unknown')
                    }
        except Exception as e:
            logger.error(f"Vector similarity search failed: {e}", exc_info=True)


        # Step 3: Batched Graph Relationship Search (if entities were found)
        if entities_in_question:
            logger.debug(f"Performing graph traversal for {len(entities_in_question)} entities")
            lower_entities = [entity.lower() for entity in entities_in_question]
            
            # Direct mentions query
            direct_query = """
            MATCH (c:Chunk)-[:MENTIONS]->(e:Entity)
            WHERE toLower(e.name) IN $entity_names
            OPTIONAL MATCH (c)<-[:CONTAINS]-(d:Document)
            RETURN c.id AS chunk_id, c.content AS content, c.page_num AS page_num,
                   d.id AS source_document, 'direct' AS type, 1.0 AS score, 
                   collect(DISTINCT e.name) AS matched_entities
            LIMIT $limit
            """
            
            try:
                direct_results = graph.query(direct_query, params={
                    "entity_names": lower_entities,
                    "limit": k_graph
                })
                
                # Process direct results
                for idx, result in enumerate(direct_results):
                    chunk_id = result.get('chunk_id')
                    if not chunk_id: continue
                    
                    if chunk_id not in contexts_dict:
                        contexts_dict[chunk_id] = {
                            "source": "graph_direct",
                            "rank": idx,
                            "score": result.get('score', 1.0),
                            "chunk_id": chunk_id,
                            "content": result.get('content', ''),
                            "page_num": result.get('page_num'),
                            "source_document": result.get('source_document', 'unknown'),
                            "matched_entities": result.get('matched_entities', [])
                        }
                    else:
                        contexts_dict[chunk_id].setdefault("matched_entities", []).extend(result.get('matched_entities', []))
                        contexts_dict[chunk_id]["matched_entities"] = list(set(contexts_dict[chunk_id]["matched_entities"]))
            
            except Exception as e:
                logger.error(f"Graph traversal failed: {e}", exc_info=True)
            
            # Related mentions query
            related_query = """
            MATCH (e1:Entity)-[:RELATED_TO]-(e2:Entity)<-[:MENTIONS]-(c:Chunk)
            WHERE toLower(e1.name) IN $entity_names AND NOT toLower(e2.name) IN $entity_names
            OPTIONAL MATCH (c)<-[:CONTAINS]-(d:Document)
            RETURN c.id AS chunk_id, c.content AS content, c.page_num AS page_num,
                   d.id AS source_document, 'related' AS type, 0.5 AS score, 
                   collect(DISTINCT e2.name) AS matched_entities,
                   collect(DISTINCT e1.name) AS query_entities
            LIMIT $limit
            """
            
            try:
                related_results = graph.query(related_query, params={
                    "entity_names": lower_entities,
                    "limit": k_graph
                })
                
                # Process related results
                for idx, result in enumerate(related_results):
                    chunk_id = result.get('chunk_id')
                    if not chunk_id: continue
                    
                    if chunk_id not in contexts_dict:
                        contexts_dict[chunk_id] = {
                            "source": "graph_related",
                            "rank": idx,
                            "score": result.get('score', 0.5),
                            "chunk_id": chunk_id,
                            "content": result.get('content', ''),
                            "page_num": result.get('page_num'),
                            "source_document": result.get('source_document', 'unknown'),
                            "matched_entities": result.get('matched_entities', [])
                        }
                    else:
                        contexts_dict[chunk_id].setdefault("matched_entities", []).extend(result.get('matched_entities', []))
                        contexts_dict[chunk_id]["matched_entities"] = list(set(contexts_dict[chunk_id]["matched_entities"]))
            
            except Exception as e:
                logger.error(f"Graph traversal failed: {e}", exc_info=True)

        # Convert dict back to list
        contexts_list = list(contexts_dict.values())

        # Optional: Sort final list by score (descending), prioritizing vector score if available?
        contexts_list.sort(key=lambda x: x.get('score', 0), reverse=True)

        logger.info(f"Context gathering completed in {time.time() - start_time:.2f} seconds. Found {len(contexts_list)} unique relevant contexts.")
        return contexts_list

    except Exception as e:
        logger.error(f"Error getting query context: {str(e)}", exc_info=True)
        return list(contexts_dict.values()) # Return whatever was collected


def answer_question(
    question: str,
    api_key: str,
    contexts: List[Dict[str, Any]],
    neo4j_uri: str,
    neo4j_username: str,
    neo4j_password: str,
    model_name: str = "deepseek-chat",  # Changed from Fireworks model
    max_context_tokens: int = 3000 # Estimate max tokens for context to avoid exceeding limit
) -> Dict[str, Any]:
    """Generate answer based on retrieved contexts using DeepSeek API."""
    logger.info(f"Generating answer for question: {question}")
    start_time = time.time()

    if not contexts:
        logger.warning("No context provided to answer the question.")
        return {
            "answer": "I couldn't find any relevant information in the knowledge base to answer this question.",
            "processing_time": time.time() - start_time,
            "contexts_used": 0,
            "context_sources": []
         }

    try:
        # --- Prepare Context ---
        # Sort contexts (e.g., by score if available, or vector first) - assuming already sorted by get_query_context
        compiled_context = ""
        current_token_count = 0 # Rough estimate
        contexts_included = 0
        contexts_details = []  # Store details about included contexts

        for ctx in contexts:
            # Format context entry - include key details like source, doc, page
            ctx_entry = f"Source Type: {ctx['source']}\n"
            if ctx.get('source_document'): ctx_entry += f"Document: {ctx['source_document']}\n"
            if ctx.get('page_num') is not None: ctx_entry += f"Page: {ctx['page_num']}\n"
            if ctx.get('matched_entities'): ctx_entry += f"Matched Entities: {', '.join(ctx['matched_entities'])}\n"
            ctx_entry += f"Content: {ctx.get('content', '')}\n\n"

            # Store details for response
            context_detail = {
                "source_type": ctx['source'],
                "document": ctx.get('source_document', 'Unknown'),
                "page": ctx.get('page_num'),
                "matched_entities": ctx.get('matched_entities', [])
            }
            contexts_details.append(context_detail)

            # Estimate token count (simple space split, adjust if using a proper tokenizer)
            entry_token_estimate = len(ctx_entry.split())
            if current_token_count + entry_token_estimate <= max_context_tokens:
                compiled_context += ctx_entry
                current_token_count += entry_token_estimate
                contexts_included += 1
            else:
                logger.warning(f"Stopping context inclusion after {contexts_included} entries due to estimated token limit ({max_context_tokens}).")
                break # Stop adding contexts if limit is reached

        if not compiled_context:
             logger.error("Context compilation resulted in empty string, possibly due to token limits or empty contexts.")
             # Handle case where even the first context is too long?
             return {"error": "Failed to compile context within token limits."}


        # --- Retrieve Graph Schema ---
        try:
            graph = Neo4jGraph(url=neo4j_uri, username=neo4j_username, password=neo4j_password)
            # graph.schema is a property that fetches the schema string
            schema_text = graph.schema
            logger.debug("Retrieved graph schema for prompt.")
        except Exception as e:
            logger.warning(f"Could not retrieve graph schema: {e}. Proceeding without it.")
            schema_text = "Schema information unavailable."

        # --- Formulate Prompt ---
        prompt = f"""You are an AI assistant answering questions based on information retrieved from a knowledge base.
Use the following context information and graph schema to answer the question accurately and concisely.
Cite the source document and page number if available and relevant using bracket notation (e.g., [Document: report.pdf, Page: 5]).
If the information is not available in the provided context, state that clearly.

**Graph Schema:**
{schema_text}

**Retrieved Context:**
{compiled_context}

**Question:**
{question}

**Answer:**
"""

        # --- Call DeepSeek LLM ---
        # Initialize OpenAI client with DeepSeek base URL
        client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com", timeout=60.0)
        
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant answering questions based on provided context."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1024, # Max tokens for the *answer*
            temperature=0.1 # Low temperature for factual answers
        )

        # --- Process Response ---
        if response.choices and len(response.choices) > 0:
            answer = response.choices[0].message.content.strip()
        else:
            raise ValueError("No response content found in DeepSeek API response.")

        # Basic cleaning (less aggressive than before)
        cleaned_answer = answer # Keep most LLM formatting unless problematic

        # Count the number of contexts from each source type
        source_type_counts = {}
        for i in range(contexts_included):
            source_type = contexts_details[i]["source_type"]
            source_type_counts[source_type] = source_type_counts.get(source_type, 0) + 1
        
        # Format the context source information for display
        context_source_details = []
        for source_type, count in source_type_counts.items():
            context_source_details.append(f"{source_type} ({count})")
        
        # Extract all unique entity matches for display
        all_entities = []
        for i in range(contexts_included):
            all_entities.extend(contexts_details[i].get("matched_entities", []))
        unique_entities = list(set(all_entities))

        result = {
            "answer": cleaned_answer,
            "processing_time": time.time() - start_time,
            "contexts_used": contexts_included,
            "context_sources": list(set([c['source'] for c in contexts[:contexts_included]])),  # Sources actually used
            "context_source_counts": source_type_counts,  # Count of each source type
            "context_details": contexts_details[:contexts_included],  # Detailed info about each context
            "matched_entities": unique_entities  # All entities matched in used contexts
        }

        logger.info(f"Answer generated in {time.time() - start_time:.2f} seconds")
        logger.info(f"Context sources used: {', '.join(context_source_details)}")
        if unique_entities:
            logger.info(f"Entities matched: {', '.join(unique_entities)}")
        return result

    except Exception as e:
        logger.error(f"Error generating answer: {str(e)}", exc_info=True)
        return {"error": f"Error generating answer: {str(e)}"}

# Example Usage (Illustrative - adapt to your main script)
# if __name__ == '__main__':
#     # --- Configuration ---
#     NEO4J_URI = "bolt://localhost:7687"
#     NEO4J_USERNAME = "neo4j"
#     NEO4J_PASSWORD = "your_password"
#     FIREWORKS_API_KEY = "your_fireworks_api_key"
#     from langchain_community.embeddings import OllamaEmbeddings # Or your chosen embedding model

#     # --- Initialization ---
#     try:
#         graph_db = Neo4jGraph(url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD)
#         embedding_model = OllamaEmbeddings(model="nomic-embed-text") # Example
#         vector_idx = Neo4jVector(
#             embedding=embedding_model,
#             url=NEO4J_URI,
#             username=NEO4J_USERNAME,
#             password=NEO4J_PASSWORD,
#             index_name="chunk_embeddings", # Match the name used in setup
#             node_label="Chunk",
#             text_node_property="content",
#             embedding_node_property="embedding"
#         )
#         logger.info("Neo4j connection and vector store initialized.")
#     except Exception as init_error:
#         logger.critical(f"Initialization failed: {init_error}", exc_info=True)
#         exit(1)

#     # --- Query ---
#     test_question = "What are the key specifications for the BK 1-WAY housing?"

#     retrieved_contexts = get_query_context(
#         question=test_question,
#         vector_store=vector_idx,
#         graph=graph_db,
#         api_key=FIREWORKS_API_KEY,
#         k_vector=3,
#         k_graph=5,
#         use_llm_extraction=True # Use LLM for entity extraction
#     )

#     print("\n--- Retrieved Contexts ---")
#     for i, ctx in enumerate(retrieved_contexts):
#         print(f"Context {i+1}:")
#         print(f"  Source: {ctx.get('source')}, Score: {ctx.get('score', 'N/A')}")
#         print(f"  Chunk ID: {ctx.get('chunk_id')}")
#         print(f"  Document: {ctx.get('source_document')}, Page: {ctx.get('page_num')}")
#         print(f"  Matched Entities: {ctx.get('matched_entities')}")
#         print(f"  Content: {ctx.get('content', '')[:150]}...") # Print snippet
#         print("-" * 10)


#     if retrieved_contexts:
#         final_answer = answer_question(
#             question=test_question,
#             api_key=FIREWORKS_API_KEY,
#             contexts=retrieved_contexts,
#             neo4j_uri=NEO4J_URI,
#             neo4j_username=NEO4J_USERNAME,
#             neo4j_password=NEO4J_PASSWORD
#         )
#         print("\n--- Final Answer ---")
#         print(f"Answer: {final_answer.get('answer', final_answer.get('error', 'No answer generated.'))}")
#         print(f"Processing Time: {final_answer.get('processing_time', 0):.2f}s")
#         print(f"Contexts Used: {final_answer.get('contexts_used', 0)}")
#         print(f"Context Sources: {final_answer.get('context_sources', [])}")
#     else:
#         print("\nNo context retrieved, skipping answer generation.")