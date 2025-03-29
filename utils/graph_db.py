# /mount/src/rag_hybrid/utils/graph_db.py

import time
from langchain_community.graphs import Neo4jGraph
# from langchain.vectorstores.neo4j_vector import Neo4jVector # Original import
from langchain_community.vectorstores import Neo4jVector # Corrected import path for newer langchain
from langchain.docstore.document import Document
import json
import re
from typing import List, Dict, Any, Tuple
# import fireworks.client as fw
import concurrent.futures
from threading import Lock
import random # For jitter in backoff
from datetime import datetime # For consistent datetime handling
# Import OpenAI client for DeepSeek API
from openai import OpenAI
import os
import streamlit as st
from dotenv import load_dotenv

# Assume logger is configured elsewhere, e.g., from logging_config
import logging
logger = logging.getLogger("graph_rag") # Use the same logger name as in logs
# Basic configuration if logger is not set up externally for testing
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def get_api_credentials():
    """Get API credentials from either .env file (local) or Streamlit secrets (cloud)"""
    # Try to load from streamlit secrets first (cloud deployment)
    try:
        config = {
            "neo4j_uri": st.secrets["NEO4J_URI"],
            "neo4j_username": st.secrets["NEO4J_USERNAME"],
            "neo4j_password": st.secrets["NEO4J_PASSWORD"],
            "deepseek_api_key": st.secrets["DEEPSEEK_API_KEY"],
            # Add other keys as needed
        }
        print("Loaded credentials from Streamlit secrets")
        return config
    except (KeyError, AttributeError):
        # Fall back to .env file (local development)
        load_dotenv()
        config = {
            "neo4j_uri": os.getenv("NEO4J_URI"),
            "neo4j_username": os.getenv("NEO4J_USERNAME"),
            "neo4j_password": os.getenv("NEO4J_PASSWORD"),
            "deepseek_api_key": os.getenv("DEEPSEEK_API_KEY"),
            # Add other keys as needed
        }
        print("Loaded credentials from .env file")
        return config

# --- connect_to_neo4j (Seems OK) ---
def connect_to_neo4j(uri: str, username: str, password: str) -> Neo4jGraph:
    """Establish connection to Neo4j and return graph object"""
    try:
        logger.info(f"Connecting to Neo4j at {uri}")
        graph = Neo4jGraph(
            url=uri,
            username=username,
            password=password
        )
        # Test connection
        result = graph.query("MATCH (n) RETURN count(n) as count")
        logger.info(f"Successfully connected to Neo4j. Database has {result[0]['count']} nodes")
        return graph
    except Exception as e:
        logger.error(f"Failed to connect to Neo4j: {str(e)}", exc_info=True)
        raise

# --- setup_neo4j_schema (Improved logging for existing items) ---
def setup_neo4j_schema(graph: Neo4jGraph) -> None:
    """Define and setup the graph schema with constraints and indexes"""
    logger.info("Setting up Neo4j schema and constraints")
    try:
        # Create constraints
        constraints = [
            "CREATE CONSTRAINT document_id IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE",
            "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE",
            "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
        ]

        for constraint in constraints:
            try:
                graph.query(constraint)
                logger.debug(f"Ensured constraint exists: {constraint.split(' IF NOT EXISTS')[0]}") # Cleaner log
            except Exception as e:
                # Check if it's the specific error code for constraint already existing if possible
                # Neo.ClientError.Schema.ConstraintAlreadyExists
                if "already exists" in str(e).lower() or (hasattr(e, 'code') and "ConstraintAlreadyExists" in e.code):
                     logger.warning(f"Constraint likely already exists (this is expected): {constraint.split(' IF NOT EXISTS')[0]}")
                else:
                     logger.error(f"Failed to apply constraint: {constraint} - {str(e)}", exc_info=True)
                     # Decide if you want to raise the error or continue

        # Create indexes (Ensure they target correct properties)
        # For Neo4j 4.x+ the syntax is different for fulltext indexes
        basic_indexes = [
            # Vector index is created separately by Neo4jVector
            "CREATE INDEX document_source_idx IF NOT EXISTS FOR (d:Document) ON (d.id)", # Index on the unique ID is good
            "CREATE INDEX entity_name_idx IF NOT EXISTS FOR (e:Entity) ON (e.name)", # Index entity names
            "CREATE INDEX entity_type_idx IF NOT EXISTS FOR (e:Entity) ON (e.type)", # Index entity types
        ]

        # Try creating basic indexes first
        for index in basic_indexes:
            try:
                graph.query(index)
                logger.debug(f"Ensured index exists: {index.split(' IF NOT EXISTS')[0]}")
            except Exception as e:
                if "already exists" in str(e).lower() or (hasattr(e, 'code') and "IndexAlreadyExists" in e.code):
                    logger.warning(f"Index likely already exists (this is expected): {index.split(' IF NOT EXISTS')[0]}")
                else:
                    logger.error(f"Failed to create index: {index} - {str(e)}", exc_info=True)

        # Now try creating fulltext index with appropriate syntax for different Neo4j versions
        try:
            # First try Neo4j 4.x+ syntax for fulltext index
            fulltext_query = """
            CALL db.index.fulltext.createNodeIndex(
                'chunk_content_fulltext_idx',  // index name
                ['Chunk'],                     // node labels
                ['content'],                   // property names
                {eventually_consistent: 'true'} // optional config
            )
            """
            graph.query(fulltext_query)
            logger.debug("Created fulltext index 'chunk_content_fulltext_idx' on Chunk.content (Neo4j 4.x syntax)")
        except Exception as e1:
            if "already exists" in str(e1).lower():
                logger.warning("Fulltext index 'chunk_content_fulltext_idx' already exists.")
            else:
                # Try alternative syntax (Neo4j 3.5 or legacy)
                try:
                    alt_fulltext_query = """
                    CREATE FULLTEXT INDEX chunk_content_fulltext_idx 
                    FOR (n:Chunk) 
                    ON EACH [n.content]
                    """
                    graph.query(alt_fulltext_query)
                    logger.debug("Created fulltext index 'chunk_content_fulltext_idx' on Chunk.content (legacy syntax)")
                except Exception as e2:
                    logger.error(f"Could not create fulltext index with any syntax: {str(e1)} / {str(e2)}", exc_info=True)
                    # Continue without fulltext index - not critical for basic operations

        logger.info("Neo4j schema setup completed")
    except Exception as e:
        logger.error(f"Error setting up Neo4j schema: {str(e)}", exc_info=True)
        raise


# --- _extract_json_from_response (More Robust Parsing) ---
def _extract_json_from_response(text: str, chunk_id: str) -> List[Dict[str, Any]]:
    """Tries to extract a JSON list from the LLM response text."""
    text = text.strip() # Remove leading/trailing whitespace

    # 1. Try finding ```json ... ``` blocks
    json_block_match = re.search(r'```(?:json)?\s*(\[.*?\])\s*```', text, re.DOTALL | re.IGNORECASE)
    if json_block_match:
        json_str = json_block_match.group(1)
        try:
            # Basic cleanup: Remove potential trailing commas before closing bracket/brace
            json_str_cleaned = re.sub(r',\s*([\}\]])', r'\1', json_str)
            return json.loads(json_str_cleaned)
        except json.JSONDecodeError as e:
            logger.warning(f"Chunk {chunk_id}: Failed to parse JSON within ```json block: {e}. Raw block: {json_str[:500]}...")
            # Fall through to next method

    # 2. Try finding the first list using non-greedy matching (if no ```json block)
    # Be careful, this might grab unrelated lists if the LLM adds conversational text
    # Let's check if the text *starts* with '[' as a heuristic
    if text.startswith('['):
        list_match = re.match(r'(\[.*?\])', text, re.DOTALL) # Match from start
        if list_match:
            json_str = list_match.group(1)
            try:
                json_str_cleaned = re.sub(r',\s*([\}\]])', r'\1', json_str)
                return json.loads(json_str_cleaned)
            except json.JSONDecodeError as e:
                logger.warning(f"Chunk {chunk_id}: Failed to parse JSON list starting the response: {e}. Raw string: {json_str[:500]}...")
                # Fall through

    # 3. Try finding the first object (if list fails, maybe it returned a single object?)
    # Check if text starts with '{'
    if text.startswith('{'):
        object_match = re.match(r'(\{.*?\})', text, re.DOTALL) # Match from start
        if object_match:
            json_str = object_match.group(1)
            try:
                json_str_cleaned = re.sub(r',\s*([\}\]])', r'\1', json_str)
                # Wrap in a list as the function expects a list
                return [json.loads(json_str_cleaned)]
            except json.JSONDecodeError as e:
                logger.warning(f"Chunk {chunk_id}: Failed to parse JSON object starting the response: {e}. Raw string: {json_str[:500]}...")

    logger.warning(f"Chunk {chunk_id}: Could not extract valid JSON list or object from response. Raw response start: {text[:500]}...")
    return []

# --- Transaction function for Neo4j (Fix Applied Here) ---
def _create_chunk_and_entities_tx(tx, chunk_id, source_doc_id, content, page_num, entities):
    """
    Neo4j transaction function to create/merge Chunk, Entities, and relationships.
    Executes within a single atomic transaction.
    """
    # Create/Merge chunk node and link to document
    # NOTE: THE FIX IS HERE - Removed the trailing "WITH c"
    chunk_query = """
    MATCH (d:Document {id: $doc_id}) // Ensure document exists before creating chunk relationship
    MERGE (c:Chunk {id: $chunk_id})
    ON CREATE SET c.content = $content,
                  c.page_num = $page_num,
                  c.source_doc_id = $doc_id, // Add doc id property for easier lookup
                  c.created_at = datetime() // Track creation time
    ON MATCH SET  c.content = $content, // Update content if chunk existed
                  c.page_num = $page_num,
                  c.source_doc_id = $doc_id,
                  c.updated_at = datetime() // Track update time
    MERGE (d)-[r_contains:CONTAINS]->(c) // Ensure relationship exists
    // No RETURN or WITH needed here, MERGE is a valid concluding clause for this tx.run part
    """
    # Run the first part of the transaction
    tx.run(chunk_query, chunk_id=chunk_id, content=content, page_num=page_num, doc_id=source_doc_id)

    # Create/Merge entity nodes and MENTIONS relationships (if entities were extracted)
    # Use UNWIND for efficient batching of entity creation/merging
    if entities: # Only run if there are entities to process
        # Make sure entities list contains valid dictionaries
        valid_entities = [e for e in entities if isinstance(e, dict) and e.get("name") and e.get("type")]
        if not valid_entities:
            logger.warning(f"Chunk {chunk_id}: Entity list provided, but contained no valid entities after filtering.")
            return # Nothing more to do in this transaction part

        entity_query = """
        UNWIND $entities as entity_data // $entities should be the list of dicts [{name:x, type:y}, ...]
        // Add WITH clause before WHERE to fix syntax error
        WITH entity_data
        WHERE entity_data.name IS NOT NULL AND entity_data.type IS NOT NULL AND trim(entity_data.name) <> ""

        // Create a more robust, case-insensitive ID using APOC if available, otherwise simple concatenation
        // Ensure APOC is installed in Neo4j plugins if using apoc.text.clean
        WITH entity_data,
             // Use coalesce to handle potential nulls if APOC isn't installed or fails
             coalesce(apoc.text.clean(toLower(entity_data.name)), toLower(trim(entity_data.name))) + '_' +
             coalesce(apoc.text.clean(toLower(entity_data.type)), toLower(trim(entity_data.type)))
             as entity_id

        MERGE (e:Entity {id: entity_id})
        ON CREATE SET e.name = entity_data.name, // Store original case name
                      e.type = entity_data.type,
                      e.created_at = datetime()
        ON MATCH SET  e.name = entity_data.name, // Update name/type case if it changes
                      e.type = entity_data.type,
                      e.updated_at = datetime()

        // Re-MATCH the chunk created in the first part of this transaction
        WITH e, $chunk_id as chunkId
        MATCH (c:Chunk {id: chunkId})

        // Create relationship between chunk and entity
        MERGE (c)-[r_mentions:MENTIONS]->(e)
        ON CREATE SET r_mentions.count = 1, r_mentions.created_at = datetime()
        ON MATCH SET r_mentions.count = coalesce(r_mentions.count, 0) + 1, r_mentions.updated_at = datetime()
        // No RETURN needed, MERGE is a valid concluding clause
        """
        tx.run(entity_query, entities=valid_entities, chunk_id=chunk_id)
    # else:
        # logger.debug(f"Chunk {chunk_id}: No entities to process.") # Optional: Log if no entities


# --- create_knowledge_graph (Updated for DeepSeek API) ---
def create_knowledge_graph(
    graph: Neo4jGraph,
    chunks: List[Document],
    source_metadata: Dict[str, Any],
    api_key: str,  # This should come from your config, not directly from os.getenv
    max_retries: int = 5,
    initial_backoff: float = 1.5,
    max_backoff: float = 60.0,
    llm_model_name: str = "deepseek-chat",
    max_concurrent_requests: int = 3  # Reduced to be more conservative
) -> None:
    """Create a knowledge graph from the document chunks with DeepSeek API."""
    logger.info("Starting knowledge graph creation")
    start_time = time.time()
    
    # Validate API key before making any calls
    if not api_key or len(api_key) < 20:  # Basic validation
        error_msg = "Invalid or missing DeepSeek API key. Check your environment variables or Streamlit secrets."
        logger.critical(error_msg)
        raise ValueError(error_msg)
    
    try:
        # Initialize OpenAI client with DeepSeek base URL
        openai_client = OpenAI(
            api_key=api_key, 
            base_url="https://api.deepseek.com",
            timeout=30.0  # Set a default timeout for all requests
        )
        
        # Verify connection with a simple API call
        try:
            logger.info("Testing DeepSeek API connection...")
            test_response = openai_client.chat.completions.create(
                model=llm_model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Hello"}
                ],
                max_tokens=10,
                temperature=0.0
            )
            logger.info("DeepSeek API connection successful")
        except Exception as conn_err:
            logger.critical(f"Failed to establish connection to DeepSeek API: {str(conn_err)}")
            raise ValueError(f"DeepSeek API connection test failed: {str(conn_err)}")
        
        # Continue with the rest of your function...

        # --- Create Source Document Nodes ---
        logger.info("Creating/updating source document nodes...")
        for source, metadata in source_metadata.items():
            doc_id = metadata.get("filename", source) # Prefer filename if available
            if not doc_id:
                 logger.warning(f"Skipping document node creation for source '{source}' due to missing filename/ID.")
                 continue

            logger.debug(f"Processing document node for: {doc_id}")
            query = """
            MERGE (d:Document {id: $id})
            ON CREATE SET d.title = $title,
                          d.pages = $pages,
                          d.processed_at = datetime($processed_at), // Use Neo4j datetime
                          d.created_at = datetime()
            ON MATCH SET  d.title = $title, // Update metadata even if node exists
                          d.pages = $pages,
                          d.processed_at = datetime($processed_at),
                          d.updated_at = datetime()
            // RETURN d.id // Optional return if needed
            """
            try:
                processed_time_str = metadata.get("processed_at", datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'))
                params = {
                    "id": doc_id,
                    "title": metadata.get("title", metadata.get("filename", "Unknown Title")), # Better title handling
                    "pages": metadata.get("pages", 0),
                    "processed_at": processed_time_str
                }
                graph.query(query, params=params)
            except Exception as e:
                logger.error(f"Failed to create/update document node for {doc_id}: {e}", exc_info=True)
                # Decide whether to continue or raise

        # --- Setup DeepSeek API ---

        logger.info(f"Using LLM model: {llm_model_name}")
        logger.info(f"Using max_concurrent_requests: {max_concurrent_requests}")

        # --- Process Chunks Concurrently ---
        logger.info(f"Processing {len(chunks)} chunks with concurrent API calls for entity extraction...")

        # Lock might not be strictly necessary if database operations are atomic per chunk
        # but can be useful if shared resources were involved. Removed for now.
        # lock = Lock()

        processed_count = 0
        total_chunks = len(chunks)
        progress_interval = max(1, total_chunks // 20) # Log progress roughly every 5%

        # --- Worker Function for ThreadPool ---
        def process_chunk_api_call(chunk) -> Tuple[str, str, List[Dict], str, int]:
            """Handles API call with retries and JSON extraction for a single chunk."""
            chunk_id = chunk.metadata.get('chunk_id')
            source_doc_id = chunk.metadata.get('source_document')
            page_num = chunk.metadata.get('page', 0) # Ensure page number is fetched

            if not chunk_id or not source_doc_id:
                logger.error(f"Chunk metadata missing 'chunk_id' or 'source_document'. Metadata: {chunk.metadata}")
                return (None, None, [], "", 0) # Return indicating failure

            try:
                logger.debug(f"Extracting entities from chunk {chunk_id} (Page: {page_num})")

                # Check if chunk content is empty
                if not chunk.page_content or not chunk.page_content.strip():
                     logger.warning(f"Chunk {chunk_id} has empty content. Skipping entity extraction.")
                     return (chunk_id, source_doc_id, [], chunk.page_content, page_num)

                # Improved prompt clarity and constraint
                entity_prompt = f"""
                Extract significant named entities (Person, Organization, Location, Product, Technology, etc.)
                and key concepts/topics from the following text.
                Return ONLY a valid JSON list of objects. Each object MUST have 'name' (string) and 'type' (string) keys.
                Use concise, common types. DO NOT include ```json markers, explanations, or any text outside the JSON list.

                TEXT:
                {chunk.page_content[:2000]}

                JSON LIST:
                """

                current_retry = 0
                backoff_time = initial_backoff
                entities = []

                while current_retry < max_retries:
                    try:
                        # Add detailed logging to troubleshoot API call
                        logger.debug(f"Attempt {current_retry+1}/{max_retries} - Making API call for chunk {chunk_id}")
                        
                        # Use DeepSeek API via OpenAI client - Let's see the exact error
                        try:
                            completion = openai_client.chat.completions.create(
                                model=llm_model_name,
                                messages=[
                                    {"role": "system", "content": "You are a helpful assistant that extracts named entities and concepts from text into JSON format."},
                                    {"role": "user", "content": entity_prompt}
                                ],
                                max_tokens=1024,
                                temperature=0.0, # Lower temperature for more deterministic extraction
                                # Add timeout to avoid hanging
                                timeout=30.0,
                            )
                            
                            response_content = completion.choices[0].message.content
                            logger.debug(f"Raw LLM response for chunk {chunk_id}: {response_content[:200]}...")

                            entities = _extract_json_from_response(response_content, chunk_id) # Pass chunk_id for logging
                            if not entities:
                                # Logged within _extract_json_from_response if parsing fails or empty
                                pass # Keep processing

                            break # Success, exit retry loop
                            
                        except Exception as detailed_error:
                            # Log the specific error
                            logger.error(f"Detailed API error for chunk {chunk_id}: {type(detailed_error).__name__}: {str(detailed_error)}")
                            raise # Re-raise to be caught by the outer handler

                    except Exception as api_err:
                        current_retry += 1
                        if current_retry >= max_retries:
                            logger.error(f"API call final failure after {max_retries} retries for chunk {chunk_id}. Skipping entity extraction for this chunk. Error: {api_err}")
                            # Return successfully processed chunk info but with empty entities
                            return (chunk_id, source_doc_id, [], chunk.page_content, page_num)
                        else:
                            # Exponential backoff with jitter
                            wait_time = backoff_time + random.uniform(0, backoff_time * 0.5)
                            wait_time = min(wait_time, max_backoff) # Ensure wait time doesn't exceed max
                            logger.warning(f"API call failed for chunk {chunk_id}. Retrying in {wait_time:.2f} seconds... (Attempt {current_retry}/{max_retries}). Error: {type(api_err).__name__}: {str(api_err)}")
                            time.sleep(wait_time)
                            backoff_time = min(backoff_time * 2, max_backoff) # Double backoff time, capped

                # Log if entities were successfully extracted or if none were found after successful API call
                if entities:
                    logger.debug(f"Successfully extracted {len(entities)} entities for chunk {chunk_id}")
                else:
                    logger.info(f"No entities found or parsed for chunk {chunk_id} after successful API call(s).")

                return (chunk_id, source_doc_id, entities, chunk.page_content, page_num)

            except Exception as e:
                logger.error(f"Unhandled error processing chunk {chunk_id} in API call worker: {str(e)}", exc_info=True)
                # Return empty entities list if processing fails catastrophically
                return (chunk_id, source_doc_id, [], chunk.page_content, page_num)


        # --- Worker Function for Storing Data in Neo4j ---
        def store_chunk_data_in_db(result_data):
            """Handles storing the processed chunk data in Neo4j using a transaction."""
            nonlocal processed_count # Allow modification of the outer scope variable
            chunk_id, source_doc_id, entities, content, page_num = result_data

            if not chunk_id or not source_doc_id:
                 # Error already logged in process_chunk_api_call if IDs were missing initially
                 logger.error(f"Attempted to store data but chunk_id or source_doc_id was missing. Skipping DB store.")
                 return # Cannot store without IDs

            try:
                # Use a single transaction for chunk + entities for atomicity
                # Get a new session for each write operation from the thread pool
                with graph._driver.session() as session:
                    session.execute_write(
                        _create_chunk_and_entities_tx, # Pass the transaction function
                        chunk_id, source_doc_id, content, page_num, entities
                    )
                logger.debug(f"Stored data for chunk {chunk_id}")

            except Exception as e:
                # Catch potential Neo4j errors (SyntaxError should be fixed, but others might occur)
                # e.g., Connection errors, constraint violations if schema changes
                logger.error(f"Error storing chunk {chunk_id} data in Neo4j transaction: {str(e)}", exc_info=True)
                # Depending on error, might implement retry here, but often DB errors need investigation

            # Update progress counter safely after DB attempt (success or failure)
            processed_count += 1
            if processed_count % progress_interval == 0 or processed_count == total_chunks:
                elapsed_time = time.time() - start_time
                logger.info(f"Processed {processed_count}/{total_chunks} chunks ({processed_count/total_chunks*100:.1f}%) in {elapsed_time:.2f} seconds")


        # --- Execute Processing and Storing using ThreadPoolExecutor ---
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent_requests, thread_name_prefix='GraphWorker') as executor:
            # Submit API call tasks
            future_to_chunk = {
                executor.submit(process_chunk_api_call, chunk): chunk
                for chunk in chunks if chunk.metadata.get('chunk_id') # Ensure chunk has an ID
            }

            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_chunk):
                original_chunk = future_to_chunk[future]
                chunk_id_for_log = original_chunk.metadata.get('chunk_id', 'UNKNOWN')
                try:
                    # Get the result from the API call worker (contains chunk info + entities)
                    api_result = future.result()
                    if api_result and api_result[0]: # Check if result is valid and has chunk_id
                        # Submit the database storage task (could use another executor or run sequentially here)
                        # Running sequentially here simplifies progress tracking and avoids overwhelming DB connection pool
                        store_chunk_data_in_db(api_result)
                    else:
                         logger.warning(f"API processing failed to return valid data for chunk {chunk_id_for_log}. Skipping DB storage.")
                         # Still increment count as we attempted to process it
                         processed_count += 1
                         if processed_count % progress_interval == 0 or processed_count == total_chunks:
                             elapsed_time = time.time() - start_time
                             logger.info(f"Processed {processed_count}/{total_chunks} chunks ({processed_count/total_chunks*100:.1f}%) in {elapsed_time:.2f} seconds")


                except Exception as exc:
                    # Catch exceptions raised from process_chunk_api_call itself if future.result() fails unexpectedly
                    logger.error(f"Error retrieving result for chunk {chunk_id_for_log}: {exc}", exc_info=True)
                    # Increment count even on failure to retrieve result
                    processed_count += 1
                    if processed_count % progress_interval == 0 or processed_count == total_chunks:
                        elapsed_time = time.time() - start_time
                        logger.info(f"Processed {processed_count}/{total_chunks} chunks ({processed_count/total_chunks*100:.1f}%) in {elapsed_time:.2f} seconds")


        logger.info(f"Completed processing all {total_chunks} chunks. Total time: {time.time() - start_time:.2f} seconds")

        # --- Create RELATED_TO relationships (Optional, can be heavy) ---
        try:
             logger.info("Creating/updating RELATED_TO relationships between co-mentioned entities...")
             related_query_start_time = time.time()
             # Optimized query: Calculate relationships based on co-occurrence within chunks
             related_query = """
             MATCH (e1:Entity)<-[:MENTIONS]-(c:Chunk)-[:MENTIONS]->(e2:Entity)
             WHERE id(e1) < id(e2) // Avoid self-loops and process pairs once
             WITH e1, e2, count(c) AS shared_chunks // Count co-mentions
             MERGE (e1)-[r:RELATED_TO]-(e2) // Use undirected merge
             ON CREATE SET r.weight = shared_chunks, r.created_at = datetime()
             ON MATCH SET r.weight = shared_chunks, r.updated_at = datetime() // Update weight based on current count
             // RETURN count(r) // Optional: return number of relationships created/updated
             """
             # Consider running in batches if graph is very large using apoc.periodic.iterate
             # For moderate graphs, a single run might be acceptable.
             graph.query(related_query)
             logger.info(f"Entity relationship update completed in {time.time() - related_query_start_time:.2f} seconds.")
        except Exception as e:
             logger.error(f"Failed to update RELATED_TO entity relationships: {str(e)}", exc_info=True)
             # This step failing might not be critical, so log error and continue

        logger.info("Knowledge graph creation process completed.")

    except Exception as e:
        logger.critical(f"Critical error during knowledge graph creation: {str(e)}", exc_info=True)
        raise # Re-raise critical errors


# --- setup_vector_index (Corrected import and parameters) ---
def setup_vector_index(
    chunks: List[Document],
    embedding_function, # Needs to be an initialized embedding model object
    neo4j_uri: str,
    neo4j_username: str,
    neo4j_password: str,
    index_name: str = "chunk_embeddings", # Configurable index name
    node_label: str = "Chunk", # Should match the node label used above
    embedding_property: str = "embedding", # Standard property name for embeddings
    text_property: str = "content" # Property containing text used for embedding
) -> Neo4jVector:
    """Setup vector embeddings in Neo4j for semantic search"""
    logger.info(f"Setting up vector index '{index_name}' on {node_label}.{embedding_property}")
    start_time = time.time()
    try:
        # Check if chunks have content and metadata required
        if not chunks:
            logger.warning("No chunks provided to setup_vector_index. Skipping.")
            return None

        valid_chunks = []
        for i, doc in enumerate(chunks):
            if not hasattr(doc, 'page_content') or not doc.page_content:
                 logger.warning(f"Chunk {i} missing page_content. Skipping for vector index.")
                 continue
            if not hasattr(doc, 'metadata') or 'chunk_id' not in doc.metadata:
                 logger.warning(f"Chunk {i} missing 'chunk_id' in metadata. Skipping for vector index. Metadata: {doc.metadata}")
                 continue
            valid_chunks.append(doc)

        if not valid_chunks:
            logger.error("No valid chunks with content and 'chunk_id' metadata found for vector index creation.")
            return None

        logger.info(f"Attempting to create/update vector index '{index_name}' with {len(valid_chunks)} valid chunks.")

        # Prepare documents with 'id' in metadata for potential matching by Neo4jVector
        documents_for_vector = []
        for doc in valid_chunks:
             new_meta = doc.metadata.copy()
             new_meta['id'] = doc.metadata['chunk_id'] # Explicitly add 'id' key
             documents_for_vector.append(Document(page_content=doc.page_content, metadata=new_meta))

        # Initialize Neo4j connection directly to set up vector index
        logger.info("Creating vector index directly via Neo4j query if it doesn't exist...")
        with Neo4jGraph(url=neo4j_uri, username=neo4j_username, password=neo4j_password) as direct_graph:
            # Check if the index exists
            index_check_query = """
            SHOW INDEXES YIELD name, type
            WHERE name = $index_name
            RETURN count(*) > 0 as exists
            """
            result = direct_graph.query(index_check_query, params={"index_name": index_name})
            index_exists = result[0]['exists'] if result else False
            
            if not index_exists:
                try:
                    # Create the vector index using direct Cypher query
                    # This is the Neo4j 5.x syntax for vector indexes
                    vector_index_query = f"""
                    CALL db.index.vector.createNodeIndex(
                      '{index_name}',
                      '{node_label}',
                      '{embedding_property}',
                      1536,
                      'cosine'
                    )
                    """
                    direct_graph.query(vector_index_query)
                    logger.info(f"Vector index '{index_name}' created successfully")
                except Exception as e:
                    # If the newer syntax fails, try alternative syntax or handle error
                    logger.error(f"Failed to create vector index with Neo4j 5.x syntax: {e}")
                    logger.info("Continuing with the expectation that Neo4jVector will handle index creation")
            else:
                logger.info(f"Vector index '{index_name}' already exists")

        logger.info("Initializing Neo4jVector store...")
        # Try initializing the store object first
        vector_store = Neo4jVector(
             embedding=embedding_function,
             url=neo4j_uri,
             username=neo4j_username,
             password=neo4j_password,
             index_name=index_name,
             node_label=node_label,
             text_node_property=text_property,
             embedding_node_property=embedding_property,
             # retrieval_query=... # Optional: define custom retrieval query
        )

        logger.info(f"Adding/updating embeddings for {len(documents_for_vector)} chunks...")
        # Use add_documents which is generally better for adding to an existing index/graph structure
        # It should use the 'id' from metadata to match nodes if configured correctly.
        vector_store.add_documents(documents_for_vector, ids=[doc.metadata['id'] for doc in documents_for_vector])

        logger.info(f"Vector index '{index_name}' setup completed for {node_label} nodes in {time.time() - start_time:.2f} seconds.")
        return vector_store # Return the initialized store object
    except Exception as e:
        logger.error(f"Error setting up vector index '{index_name}': {str(e)}", exc_info=True)
        raise


# --- get_graph_statistics (Seems OK) ---
def get_graph_statistics(graph: Neo4jGraph) -> Dict[str, int]:
    """Get basic statistics about the knowledge graph"""
    logger.info("Fetching graph statistics...")
    stats = {
        "documents": 0, "chunks": 0, "entities": 0, "relationships": 0
    }
    try:
        doc_count_res = graph.query("MATCH (d:Document) RETURN count(d) as count")
        if doc_count_res: stats["documents"] = doc_count_res[0]['count']

        chunk_count_res = graph.query("MATCH (c:Chunk) RETURN count(c) as count")
        if chunk_count_res: stats["chunks"] = chunk_count_res[0]['count']

        entity_count_res = graph.query("MATCH (e:Entity) RETURN count(e) as count")
        if entity_count_res: stats["entities"] = entity_count_res[0]['count']

        rel_count_res = graph.query("MATCH ()-[r]->() RETURN count(r) as count")
        if rel_count_res: stats["relationships"] = rel_count_res[0]['count']

        logger.info(f"Graph Statistics: {stats}")
        return stats
    except Exception as e:
        logger.error(f"Error getting graph statistics: {str(e)}", exc_info=True)
        return stats # Return default stats on error


# --- reset_neo4j_database (Improved cleanup) ---
def reset_neo4j_database(graph: Neo4jGraph) -> None:
    """Reset the Neo4j database by deleting all nodes, relationships, indexes, and constraints"""
    logger.warning("!!! Resetting Neo4j database - DELETING ALL DATA, INDEXES, AND CONSTRAINTS !!!")
    start_time = time.time()
    try:
        # 1. Drop all constraints
        logger.info("Dropping all constraints...")
        constraints = graph.query("SHOW CONSTRAINTS YIELD name")
        # Handle potential differences in SHOW CONSTRAINTS output across versions
        constraint_names = [c.get('name') for c in constraints if c.get('name')]
        if not constraint_names:
             logger.info("No constraints found to drop.")
        else:
             for name in constraint_names:
                 try:
                      logger.debug(f"Dropping constraint: {name}")
                      graph.query(f"DROP CONSTRAINT {name}")
                 except Exception as e:
                      # Ignore errors if constraint doesn't exist (might happen in race conditions or older Neo4j versions)
                      if "constraint not found" in str(e).lower():
                          logger.warning(f"Constraint '{name}' not found, likely already dropped.")
                      else:
                          logger.error(f"Could not drop constraint {name}: {e}")

        # 2. Drop all indexes (including vector and fulltext)
        logger.info("Dropping all indexes...")
        indexes = graph.query("SHOW INDEXES YIELD name, type")
        index_names = [idx.get('name') for idx in indexes if idx.get('name') and idx.get('type') != 'LOOKUP' and not idx.get('name', '').startswith('token_')]
        if not index_names:
             logger.info("No user-defined indexes found to drop.")
        else:
             for name in index_names:
                 try:
                      logger.debug(f"Dropping index: {name}")
                      graph.query(f"DROP INDEX {name}")
                 except Exception as e:
                      if "index not found" in str(e).lower():
                           logger.warning(f"Index '{name}' not found, likely already dropped.")
                      else:
                           logger.error(f"Could not drop index {name}: {e}")

        # 3. Delete all nodes and relationships (Batched)
        logger.info("Deleting nodes and relationships in batches...")
        total_deleted = 0
        batch_size = 10000 # Adjust batch size based on memory/performance
        while True:
            # Delete a batch of nodes and their relationships atomically
            result = graph.query(f"MATCH (n) WITH n LIMIT {batch_size} DETACH DELETE n RETURN count(n) as deleted_count")
            deleted_count = result[0]['deleted_count'] if result else 0
            total_deleted += deleted_count
            if deleted_count == 0:
                logger.info("No more nodes to delete.")
                break # No more nodes to delete
            logger.info(f"Deleted {deleted_count} nodes in this batch (Total: {total_deleted})...")
            # Optional: Add a small sleep if deletion puts heavy load on the DB
            # time.sleep(0.05)

        # 4. Verify database is empty
        result = graph.query("MATCH (n) RETURN count(n) as count")
        node_count = result[0]['count'] if result else -1
        rel_result = graph.query("MATCH ()-->() RETURN count(*) as count") # Check relationships too
        rel_count = rel_result[0]['count'] if rel_result else -1

        logger.info(f"Database reset complete in {time.time() - start_time:.2f} seconds.")
        if node_count == 0 and rel_count == 0:
            logger.info("Database is now empty (0 nodes, 0 relationships).")
        else:
            logger.warning(f"Database reset might be incomplete: {node_count} nodes, {rel_count} relationships remain.")

    except Exception as e:
        logger.critical(f"Failed to reset Neo4j database: {str(e)}", exc_info=True)
        raise