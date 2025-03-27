import time
from langchain_community.graphs import Neo4jGraph
# from langchain.vectorstores.neo4j_vector import Neo4jVector # Original import
from langchain_community.vectorstores import Neo4jVector # Corrected import path for newer langchain
from langchain.docstore.document import Document
import json
import re
from typing import List, Dict, Any, Tuple
import fireworks.client as fw
import concurrent.futures
from threading import Lock
import random # For jitter in backoff

# Import the specific error type
from fireworks.client.error import RateLimitError

from utils.logging_config import logger

# --- connect_to_neo4j (no changes needed) ---
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

# --- setup_neo4j_schema (no changes needed) ---
def setup_neo4j_schema(graph: Neo4jGraph) -> None:
    """Define and setup the graph schema with constraints and indexes"""
    logger.info("Setting up Neo4j schema and constraints")
    try:
        # Create constraints
        constraints = [
            "CREATE CONSTRAINT document_id IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE",
            "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE",
            "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
            # "CREATE CONSTRAINT concept_id IF NOT EXISTS FOR (c:Concept) REQUIRE c.id IS UNIQUE", # Removed if Concept label isn't used
            # "CREATE CONSTRAINT relationship_type IF NOT EXISTS FOR ()-[r:MENTIONS]-() REQUIRE r.type IS NOT NULL", # Usually not needed/causes issues if relationship doesn't always have type
        ]

        for constraint in constraints:
            try:
                graph.query(constraint)
                logger.debug(f"Applied constraint: {constraint}")
            except Exception as e:
                # Be more specific about expected "already exists" errors if possible
                if "already exists" in str(e).lower():
                     logger.warning(f"Constraint likely already exists: {constraint}")
                else:
                     logger.error(f"Failed to apply constraint: {constraint} - {str(e)}", exc_info=True)


        # Create indexes (Ensure they target correct properties)
        indexes = [
            "CREATE INDEX document_source_idx IF NOT EXISTS FOR (d:Document) ON (d.id)", # Index on the unique ID is good
            "CREATE FULLTEXT INDEX chunk_content_fulltext_idx IF NOT EXISTS FOR (c:Chunk) ON (c.content)", # Use FULLTEXT for text search
            "CREATE INDEX entity_name_idx IF NOT EXISTS FOR (e:Entity) ON (e.name)", # Index entity names
            "CREATE INDEX entity_type_idx IF NOT EXISTS FOR (e:Entity) ON (e.type)", # Index entity types
        ]

        for index in indexes:
            try:
                graph.query(index)
                logger.debug(f"Created index: {index}")
            except Exception as e:
                if "already exists" in str(e).lower():
                    logger.warning(f"Index likely already exists: {index}")
                else:
                    logger.error(f"Failed to create index: {index} - {str(e)}", exc_info=True)


        logger.info("Neo4j schema setup completed")
    except Exception as e:
        logger.error(f"Error setting up Neo4j schema: {str(e)}", exc_info=True)
        raise


def _extract_json_from_response(text: str) -> List[Dict[str, Any]]:
    """Tries to extract a JSON list from the LLM response text."""
    # 1. Try finding ```json ... ``` blocks
    json_block_match = re.search(r'```json\s*(\[.*?\])\s*```', text, re.DOTALL | re.IGNORECASE)
    if json_block_match:
        json_str = json_block_match.group(1)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON within ```json block: {e}")
            # Fall through to next method

    # 2. Try finding the first list using non-greedy matching
    list_match = re.search(r'(\[.*?\])', text, re.DOTALL)
    if list_match:
        json_str = list_match.group(1)
        try:
            # Basic cleanup: Remove potential trailing commas before closing bracket/brace
            json_str = re.sub(r',\s*([\}\]])', r'\1', json_str)
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse first JSON list found: {e}")
            # Fall through to next method

    # 3. Try finding the first object (if list fails, maybe it returned a single object?)
    object_match = re.search(r'(\{.*?\})', text, re.DOTALL)
    if object_match:
        json_str = object_match.group(1)
        try:
            # Basic cleanup
            json_str = re.sub(r',\s*([\}\]])', r'\1', json_str)
            # Wrap in a list as the function expects a list
            return [json.loads(json_str)]
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse first JSON object found: {e}")

    logger.warning("Could not extract valid JSON list or object from response.")
    return []


def create_knowledge_graph(
    graph: Neo4jGraph,
    chunks: List[Document],
    source_metadata: Dict[str, Any],
    api_key: str,
    max_retries: int = 5, # Max retries for API calls
    initial_backoff: float = 1.0, # Initial wait time in seconds
    max_backoff: float = 60.0, # Maximum wait time
) -> None:
    """Create a knowledge graph from the document chunks with rate limit handling."""
    logger.info("Starting knowledge graph creation")
    try:
        # Create nodes for source documents
        for source, metadata in source_metadata.items():
            logger.debug(f"Creating document node for {source}")
            # Use MERGE to avoid duplicates, ensure `filename` is the unique identifier
            doc_id = metadata.get("filename", source) # Prefer filename if available
            query = """
            MERGE (d:Document {id: $id})
            ON CREATE SET d.title = $title,
                          d.pages = $pages,
                          d.processed_at = datetime($processed_at) // Store as Neo4j datetime
            ON MATCH SET  d.title = $title, // Update metadata even if node exists
                          d.pages = $pages,
                          d.processed_at = datetime($processed_at)
            RETURN d
            """
            params = {
                "id": doc_id,
                "title": metadata.get("filename", "Unknown Title"),
                "pages": metadata.get("pages", 0),
                # Ensure processed_at is in ISO 8601 format or convertible
                "processed_at": metadata.get("processed_at", time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
            }
            graph.query(query, params=params)

        # Setup Fireworks API key
        fw.api_key = api_key

        # Reduce max_concurrent_requests to be safer with rate limits
        # Start lower (e.g., 5-8) and increase if stable
        max_concurrent_requests = 5 # ADJUST THIS based on testing and Fireworks plan
        logger.info(f"Using max_concurrent_requests: {max_concurrent_requests}")

        lock = Lock()

        logger.info(f"Processing {len(chunks)} chunks with concurrent API calls")

        def process_chunk(chunk) -> Tuple[str, str, List[Dict], str, int]:
            chunk_id = chunk.metadata['chunk_id']
            source_doc_id = chunk.metadata['source_document']
            page_num = chunk.metadata.get('page', 0) # Ensure page number is fetched

            try:
                logger.debug(f"Extracting entities from chunk {chunk_id}")

                # Slightly improved prompt clarity
                entity_prompt = f"""
                Extract all significant named entities (like people, organizations, locations, products, technologies)
                and key concepts or topics from the following text.
                Return the results STRICTLY as a JSON list of objects. Each object must have a "name" (string)
                and a "type" (string) property. Use concise and common type names (e.g., Person, Organization, Location, Technology, Concept, Topic).
                Do NOT include any explanations or introductory text outside the JSON list itself.

                TEXT:
                {chunk.page_content[:1500]}

                JSON RESPONSE:
                """

                current_retry = 0
                backoff_time = initial_backoff
                entities = []

                while current_retry < max_retries:
                    try:
                        entity_response = fw.ChatCompletion.create(
                            model="accounts/fireworks/models/llama-v3p1-8b-instruct",
                            messages=[{"role": "user", "content": entity_prompt}],
                            max_tokens=1024,
                            temperature=0.1,
                            # Add response_format if supported by model/API for guaranteed JSON
                            # response_format={"type": "json_object"}, # Check Fireworks documentation if this is supported
                        )

                        response_content = entity_response.choices[0].message.content
                        logger.debug(f"Raw response for chunk {chunk_id}: {response_content[:200]}...") # Log start of response

                        entities = _extract_json_from_response(response_content)
                        if not entities:
                             logger.warning(f"No entities extracted or parsed for chunk {chunk_id}")

                        break # Success, exit retry loop

                    except RateLimitError as rle:
                        current_retry += 1
                        if current_retry >= max_retries:
                            logger.error(f"Rate limit exceeded after {max_retries} retries for chunk {chunk_id}. Skipping. Error: {rle}")
                            raise # Re-raise the last error if all retries fail
                        else:
                            # Exponential backoff with jitter
                            wait_time = backoff_time + random.uniform(0, backoff_time * 0.5)
                            logger.warning(f"Rate limit hit for chunk {chunk_id}. Retrying in {wait_time:.2f} seconds... (Attempt {current_retry}/{max_retries})")
                            time.sleep(wait_time)
                            backoff_time = min(backoff_time * 2, max_backoff) # Double backoff time, capped

                    except Exception as api_err:
                        # Handle other potential API errors
                        logger.error(f"API call failed for chunk {chunk_id}: {api_err}", exc_info=True)
                        # Depending on the error, you might want to retry or just fail
                        raise api_err # Re-raise for now

                return (chunk_id, source_doc_id, entities, chunk.page_content, page_num)

            except Exception as e:
                logger.error(f"Error processing chunk {chunk_id}: {str(e)}", exc_info=True)
                # Return empty entities list if processing fails
                return (chunk_id, source_doc_id, [], chunk.page_content, page_num)

        def store_chunk_data(result):
            chunk_id, source_doc_id, entities, content, page_num = result

            if not chunk_id or not source_doc_id:
                 logger.error(f"Missing chunk_id or source_doc_id in result: {result}")
                 return # Cannot store without IDs

            try:
                # Use a single transaction for chunk + entities for atomicity
                with graph._driver.session() as session:
                    session.execute_write(
                        _create_chunk_and_entities_tx,
                        chunk_id, source_doc_id, content, page_num, entities
                    )
                logger.debug(f"Stored data for chunk {chunk_id}")

            except Exception as e:
                logger.error(f"Error storing chunk {chunk_id} data in transaction: {str(e)}", exc_info=True)

        # Transaction function for Neo4j (avoids locking issues with Langchain's graph.query)
        def _create_chunk_and_entities_tx(tx, chunk_id, source_doc_id, content, page_num, entities):
            # Create chunk node and link to document
            chunk_query = """
            MATCH (d:Document {id: $doc_id}) // Ensure document exists
            MERGE (c:Chunk {id: $chunk_id})
            ON CREATE SET c.content = $content,
                          c.page_num = $page_num,
                          c.source_doc_id = $doc_id // Add doc id property for easier lookup
            ON MATCH SET  c.content = $content, // Update content if chunk existed
                          c.page_num = $page_num,
                          c.source_doc_id = $doc_id
            MERGE (d)-[:CONTAINS]->(c)
            WITH c // Pass chunk to the next part
            """
            tx.run(chunk_query, chunk_id=chunk_id, content=content, page_num=page_num, doc_id=source_doc_id)

            # Create entity nodes and relationships
            # Use UNWIND for batching entity creation/merging
            if entities: # Only run if there are entities
                entity_query = """
                UNWIND $entities as entity_data
                // Validate entity data structure within Cypher
                WHERE entity_data.name IS NOT NULL AND entity_data.type IS NOT NULL AND entity_data.name <> ''
                WITH entity_data,
                     // Create a cleaner ID, handle potential variations
                     apoc.text.clean(toLower(entity_data.name)) + '_' + apoc.text.clean(toLower(entity_data.type)) as entity_id
                MERGE (e:Entity {id: entity_id})
                ON CREATE SET e.name = entity_data.name, e.type = entity_data.type
                ON MATCH SET  e.name = entity_data.name, e.type = entity_data.type // Ensure name/type are up-to-date
                WITH e, $chunk_id as chunkId // Pass entity and chunk_id
                MATCH (c:Chunk {id: chunkId})
                MERGE (c)-[r:MENTIONS]->(e)
                ON CREATE SET r.count = 1 // Keep track of mention count per chunk?
                ON MATCH SET r.count = coalesce(r.count, 0) + 1
                // Optionally set type on relationship if needed, but often redundant with entity type
                // SET r.type = e.type
                """
                tx.run(entity_query, entities=entities, chunk_id=chunk_id)


        # Process chunks using ThreadPoolExecutor
        progress_interval = max(1, len(chunks) // 20) # Log progress more frequently
        completed = 0

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent_requests) as executor:
            future_to_chunk_index = {executor.submit(process_chunk, chunk): i for i, chunk in enumerate(chunks)}

            for future in concurrent.futures.as_completed(future_to_chunk_index):
                chunk_index = future_to_chunk_index[future]
                try:
                    result = future.result()
                    if result: # Ensure result is not None
                        store_chunk_data(result)
                    else:
                         logger.warning(f"Processing returned None for chunk index {chunk_index}")

                    completed += 1
                    if completed % progress_interval == 0 or completed == len(chunks):
                        logger.info(f"Processed {completed}/{len(chunks)} chunks ({completed/len(chunks)*100:.1f}%)")

                except Exception as e:
                    # Catch exceptions raised from process_chunk (like final RateLimitError or API errors)
                    logger.error(f"Failed to process or store result for chunk index {chunk_index}: {str(e)}", exc_info=False) # Keep log cleaner

        logger.info(f"Completed processing all {len(chunks)} chunks")

        # Create connections between related entities (Consider running this less frequently or optimizing)
        logger.info("Creating/updating connections between related entities (co-mentioned in same chunk)")
        # Optimized query: Calculate relationships based on co-occurrence within chunks
        # This might still be heavy for large graphs. Consider if it's truly needed or can be done differently.
        try:
             related_query = """
             MATCH (e1:Entity)<-[:MENTIONS]-(c:Chunk)-[:MENTIONS]->(e2:Entity)
             WHERE id(e1) < id(e2) // Avoid self-loops and process pairs once
             MERGE (e1)-[r:RELATED_TO]-(e2) // Use undirected merge
             ON CREATE SET r.weight = 1
             ON MATCH SET r.weight = r.weight + 1
             """
             # Run in batches if needed for very large graphs
             graph.query(related_query)
             logger.info("Entity relationship weights updated.")
        except Exception as e:
             logger.error(f"Failed to update entity relationships: {str(e)}", exc_info=True)


        logger.info("Knowledge graph creation completed")
    except Exception as e:
        logger.error(f"Critical error during knowledge graph creation: {str(e)}", exc_info=True)
        raise

# --- setup_vector_index (Corrected import) ---
def setup_vector_index(
    chunks: List[Document],
    embedding_function,
    neo4j_uri: str,
    neo4j_username: str,
    neo4j_password: str,
    index_name: str = "chunk_embeddings", # Make index name configurable
    node_label: str = "Chunk",
    embedding_property: str = "embedding", # Standard property name
    text_property: str = "content" # Property containing text used for embedding
) -> Neo4jVector:
    """Setup vector embeddings in Neo4j for semantic search"""
    logger.info(f"Setting up vector index '{index_name}' on {node_label}.{embedding_property}")
    try:
        # Check if chunks have content and metadata
        if not chunks:
            logger.warning("No chunks provided to setup_vector_index. Skipping.")
            return None
        if not all(hasattr(doc, 'page_content') and hasattr(doc, 'metadata') for doc in chunks):
             logger.error("Provided documents for vector index are missing page_content or metadata.")
             raise ValueError("Invalid document structure for vector index creation.")

        texts = [doc.page_content for doc in chunks]
        # Ensure metadata contains at least an 'id' matching the Chunk node ID
        metadatas = []
        for doc in chunks:
            meta = doc.metadata.copy()
            if 'chunk_id' not in meta:
                 logger.error(f"Chunk metadata missing 'chunk_id': {meta}. Cannot link embedding.")
                 raise ValueError("Chunk metadata must contain 'chunk_id'.")
            # Neo4jVector expects 'id' in metadata to link to the node's primary key
            meta['id'] = meta['chunk_id']
            metadatas.append(meta)

        logger.info(f"Creating or updating vector index '{index_name}' with {len(texts)} texts.")

        # Use from_documents for clarity if metadata association is correct
        vector_index = Neo4jVector.from_documents(
            documents=chunks, # Pass original documents if metadata mapping is handled internally
            embedding=embedding_function,
            url=neo4j_uri,
            username=neo4j_username,
            password=neo4j_password,
            index_name=index_name,
            node_label=node_label,
            text_node_property=text_property, # Specify the text property
            embedding_node_property=embedding_property, # Specify the embedding property
            # Use id from metadata['chunk_id'] to match existing nodes
            # This relies on Neo4jVector correctly using 'chunk_id' via 'id' mapping for MERGE
        )

        # Alternate using from_texts if from_documents causes issues
        # vector_index = Neo4jVector.from_texts(
        #     texts=texts,
        #     embedding=embedding_function,
        #     metadatas=metadatas, # Ensure metadatas has 'id' key mapped to chunk_id
        #     url=neo4j_uri,
        #     username=neo4j_username,
        #     password=neo4j_password,
        #     index_name=index_name,
        #     node_label=node_label,
        #     text_node_property=text_property,
        #     embedding_node_property=embedding_property,
        #     # Assuming from_texts uses metadata 'id' to find the node
        # )

        logger.info(f"Vector index '{index_name}' setup completed for {node_label} nodes.")
        return vector_index
    except Exception as e:
        logger.error(f"Error setting up vector index '{index_name}': {str(e)}", exc_info=True)
        raise


# --- get_graph_statistics (no changes needed) ---
def get_graph_statistics(graph: Neo4jGraph) -> Dict[str, int]:
    """Get basic statistics about the knowledge graph"""
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


# --- reset_neo4j_database (minor logging improvement) ---
def reset_neo4j_database(graph: Neo4jGraph) -> None:
    """Reset the Neo4j database by deleting all nodes and relationships"""
    logger.warning("!!! Resetting Neo4j database - DELETING ALL NODES AND RELATIONSHIPS !!!")
    try:
        # Delete all constraints first (indexes are often dropped automatically)
        constraints = graph.query("SHOW CONSTRAINTS")
        for constraint in constraints:
             try:
                  graph.query(f"DROP CONSTRAINT {constraint['name']}")
                  logger.debug(f"Dropped constraint: {constraint['name']}")
             except Exception as e:
                  logger.warning(f"Could not drop constraint {constraint['name']}: {e}")

        # Delete all indexes
        indexes = graph.query("SHOW INDEXES")
        for index in indexes:
             # Avoid dropping internal/system indexes if any
             if index['type'] != 'LOOKUP' and not index['name'].startswith('token_'):
                  try:
                       graph.query(f"DROP INDEX {index['name']}")
                       logger.debug(f"Dropped index: {index['name']}")
                  except Exception as e:
                       logger.warning(f"Could not drop index {index['name']}: {e}")

        # Delete all nodes and relationships (Batched for potentially large dbs)
        logger.info("Deleting nodes and relationships in batches...")
        while True:
            # Delete a batch of nodes and their relationships
            result = graph.query("MATCH (n) WITH n LIMIT 10000 DETACH DELETE n RETURN count(n) as deleted_count")
            deleted_count = result[0]['deleted_count'] if result else 0
            if deleted_count == 0:
                break # No more nodes to delete
            logger.info(f"Deleted {deleted_count} nodes in this batch...")
            time.sleep(0.1) # Small delay between batches

        # Verify database is empty
        result = graph.query("MATCH (n) RETURN count(n) as count")
        node_count = result[0]['count'] if result else -1

        logger.info(f"Database reset complete. Database now has {node_count} nodes.")
        if node_count != 0:
            logger.warning("Database reset might be incomplete, node count is not zero.")

    except Exception as e:
        logger.error(f"Failed to reset Neo4j database: {str(e)}", exc_info=True)
        raise