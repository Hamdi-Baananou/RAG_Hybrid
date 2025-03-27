import time
from langchain_community.graphs import Neo4jGraph
from langchain.vectorstores.neo4j_vector import Neo4jVector
from langchain.docstore.document import Document
import json
import re
from typing import List, Dict, Any, Tuple
import fireworks.client as fw
import concurrent.futures
from threading import Lock

from utils.logging_config import logger

def connect_to_neo4j(uri: str, username: str, password: str) -> Neo4jGraph:
    """Establish connection to Neo4j and return graph object
    
    Args:
        uri: Neo4j connection URI
        username: Neo4j username
        password: Neo4j password
        
    Returns:
        Neo4jGraph object with active connection
    """
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

def setup_neo4j_schema(graph: Neo4jGraph) -> None:
    """Define and setup the graph schema with constraints and indexes
    
    Args:
        graph: Neo4jGraph object with active connection
    """
    logger.info("Setting up Neo4j schema and constraints")
    try:
        # Create constraints
        constraints = [
            "CREATE CONSTRAINT document_id IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE",
            "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE",
            "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
            "CREATE CONSTRAINT concept_id IF NOT EXISTS FOR (c:Concept) REQUIRE c.id IS UNIQUE",
            "CREATE CONSTRAINT relationship_type IF NOT EXISTS FOR ()-[r:MENTIONS]-() REQUIRE r.type IS NOT NULL",
        ]

        for constraint in constraints:
            try:
                graph.query(constraint)
                logger.debug(f"Applied constraint: {constraint}")
            except Exception as e:
                logger.warning(f"Constraint already exists or failed: {str(e)}")
                continue

        # Create indexes
        indexes = [
            "CREATE INDEX document_source_idx IF NOT EXISTS FOR (d:Document) ON (d.source)",
            "CREATE INDEX chunk_content_idx IF NOT EXISTS FOR (c:Chunk) ON (c.content)"
        ]

        for index in indexes:
            try:
                graph.query(index)
                logger.debug(f"Created index: {index}")
            except Exception as e:
                logger.warning(f"Index already exists or failed: {str(e)}")
                continue

        logger.info("Neo4j schema setup completed")
    except Exception as e:
        logger.error(f"Error setting up Neo4j schema: {str(e)}", exc_info=True)
        raise

def create_knowledge_graph(
    graph: Neo4jGraph, 
    chunks: List[Document], 
    source_metadata: Dict[str, Any], 
    api_key: str
) -> None:
    """Create a knowledge graph from the document chunks
    
    Args:
        graph: Neo4jGraph object with active connection
        chunks: List of document chunks
        source_metadata: Dictionary with source document metadata
        api_key: Fireworks API key for entity extraction
    """
    logger.info("Starting knowledge graph creation")
    try:
        # Create nodes for source documents
        for source, metadata in source_metadata.items():
            # Create document node
            logger.debug(f"Creating document node for {source}")
            query = """
            MERGE (d:Document {id: $id})
            SET d.title = $title,
                d.pages = $pages,
                d.processed_at = $processed_at
            RETURN d
            """
            params = {
                "id": metadata["filename"],
                "title": metadata["filename"],
                "pages": metadata["pages"],
                "processed_at": metadata["processed_at"]
            }
            graph.query(query, params=params)

        # Setup Fireworks client
        fw.client.api_key = api_key
        
        # Create a semaphore to limit concurrent API calls (adjust number as needed)
        max_concurrent_requests = 20  # Adjust based on your rate limit
        
        # Create a lock for thread-safe operations
        lock = Lock()
        
        # Process chunks in parallel
        logger.info(f"Processing {len(chunks)} chunks with concurrent API calls")
        
        # Function to process a single chunk
        def process_chunk(chunk) -> Tuple[str, str, List[Dict]]:
            chunk_id = chunk.metadata['chunk_id']
            source_doc = chunk.metadata['source_document']
            
            try:
                logger.debug(f"Extracting entities from chunk {chunk_id}")
                
                entity_prompt = f"""
                Extract all important named entities, concepts, and topics from this text.
                Return them as a JSON list of objects with "type" and "name" properties.
                Examples of entity types: Person, Organization, Product, Technology, Concept, Topic.

                TEXT: {chunk.page_content[:1500]}

                JSON RESPONSE:
                """
                
                # Using Fireworks API for completion
                entity_response = fw.ChatCompletion.create(
                    model="accounts/fireworks/models/llama-v3-70b-instruct",  # Update with appropriate model
                    messages=[{"role": "user", "content": entity_prompt}],
                    max_tokens=1024,
                    temperature=0.1
                )
                
                # Extract content from Fireworks response format
                response_content = entity_response.choices[0].message.content
                
                # Parse the JSON response
                entities = []
                try:
                    # Find JSON in the response using regex
                    json_match = re.search(r'\[.*\]', response_content, re.DOTALL)
                    if json_match:
                        entities = json.loads(json_match.group(0))
                    else:
                        # Try to find JSON with curly braces
                        json_match = re.search(r'\{.*\}', response_content, re.DOTALL)
                        if json_match:
                            potential_json = json_match.group(0)
                            entities = json.loads(f"[{potential_json}]")
                        else:
                            logger.warning(f"Could not extract JSON from entity response for chunk {chunk_id}")
                except Exception as e:
                    logger.warning(f"Failed to parse entity JSON for chunk {chunk_id}: {str(e)}")
                
                return (chunk_id, source_doc, entities, chunk.page_content, chunk.metadata.get('page', 0))
            
            except Exception as e:
                logger.error(f"Error processing chunk {chunk_id}: {str(e)}", exc_info=True)
                return (chunk_id, source_doc, [], chunk.page_content, chunk.metadata.get('page', 0))
        
        # Process chunk results and store in Neo4j
        def store_chunk_data(result):
            chunk_id, source_doc, entities, content, page_num = result
            
            try:
                # Create chunk node with its content and link to document
                with lock:  # Use lock for thread-safe database operations
                    logger.debug(f"Creating chunk node {chunk_id} and linking to document {source_doc}")
                    query = """
                    MERGE (c:Chunk {id: $chunk_id})
                    SET c.content = $content,
                        c.page_num = $page_num
                    WITH c
                    MATCH (d:Document {id: $doc_id})
                    MERGE (d)-[:CONTAINS]->(c)
                    RETURN c
                    """
                    params = {
                        "chunk_id": chunk_id,
                        "content": content,
                        "page_num": page_num,
                        "doc_id": source_doc
                    }
                    graph.query(query, params=params)
                
                    # Create entity nodes and relationships
                    for entity in entities:
                        if not isinstance(entity, dict) or 'name' not in entity or 'type' not in entity:
                            logger.warning(f"Invalid entity format: {entity}")
                            continue

                        entity_name = entity.get('name')
                        entity_type = entity.get('type')

                        if not entity_name or not entity_type:
                            continue

                        # Clean entity name and create ID
                        clean_name = re.sub(r'[^\w]', '_', entity_name).lower()
                        entity_id = f"{clean_name}_{entity_type.lower()}"

                        # Create entity node and link to chunk
                        logger.debug(f"Creating entity node {entity_id} and linking to chunk {chunk_id}")
                        query = """
                        MERGE (e:Entity {id: $entity_id})
                        SET e.name = $name,
                            e.type = $type
                        WITH e
                        MATCH (c:Chunk {id: $chunk_id})
                        MERGE (c)-[:MENTIONS {type: $type}]->(e)
                        RETURN e
                        """
                        params = {
                            "entity_id": entity_id,
                            "name": entity_name,
                            "type": entity_type,
                            "chunk_id": chunk_id
                        }
                        graph.query(query, params=params)
            except Exception as e:
                logger.error(f"Error storing chunk {chunk_id} data: {str(e)}", exc_info=True)
        
        # Use ThreadPoolExecutor to process chunks in parallel
        progress_interval = max(1, len(chunks) // 10)  # Log progress at 10% intervals
        completed = 0
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent_requests) as executor:
            # Submit all chunks for processing
            future_to_chunk = {executor.submit(process_chunk, chunk): i for i, chunk in enumerate(chunks)}
            
            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_chunk):
                chunk_index = future_to_chunk[future]
                try:
                    result = future.result()
                    store_chunk_data(result)
                    
                    # Update progress counter
                    completed += 1
                    if completed % progress_interval == 0:
                        logger.info(f"Processed {completed}/{len(chunks)} chunks ({completed/len(chunks)*100:.1f}%)")
                        
                except Exception as e:
                    logger.error(f"Error processing chunk {chunk_index}: {str(e)}", exc_info=True)
        
        logger.info(f"Completed processing all {len(chunks)} chunks")

        # Create connections between related entities
        logger.info("Creating connections between related entities")
        query = """
        MATCH (c:Chunk)-[:MENTIONS]->(e1:Entity)
        MATCH (c)-[:MENTIONS]->(e2:Entity)
        WHERE e1 <> e2
        MERGE (e1)-[r:RELATED_TO]->(e2)
        ON CREATE SET r.weight = 1
        ON MATCH SET r.weight = r.weight + 1
        """
        graph.query(query)

        logger.info("Knowledge graph creation completed")
    except Exception as e:
        logger.error(f"Error creating knowledge graph: {str(e)}", exc_info=True)
        raise

def setup_vector_index(
    chunks: List[Document], 
    embedding_function, 
    neo4j_uri: str, 
    neo4j_username: str, 
    neo4j_password: str
) -> Neo4jVector:
    """Setup vector embeddings in Neo4j for semantic search
    
    Args:
        chunks: List of document chunks
        embedding_function: Function to generate embeddings
        neo4j_uri: Neo4j connection URI
        neo4j_username: Neo4j username
        neo4j_password: Neo4j password
        
    Returns:
        Neo4jVector object for vector similarity search
    """
    logger.info("Setting up vector index for semantic search")
    try:
        # Create vector index in Neo4j with minimal parameters
        texts = [doc.page_content for doc in chunks]
        metadatas = [doc.metadata for doc in chunks]

        vector_index = Neo4jVector.from_texts(
            texts=texts,
            embedding=embedding_function,
            metadatas=metadatas,
            url=neo4j_uri,
            username=neo4j_username,
            password=neo4j_password,
            index_name="chunk_embeddings",
            node_label="Chunk"
        )

        logger.info(f"Vector index created with {len(chunks)} chunk embeddings")
        return vector_index
    except Exception as e:
        logger.error(f"Error setting up vector index: {str(e)}", exc_info=True)
        raise

def get_graph_statistics(graph: Neo4jGraph) -> Dict[str, int]:
    """Get basic statistics about the knowledge graph
    
    Args:
        graph: Neo4jGraph object with active connection
        
    Returns:
        Dictionary with count statistics
    """
    try:
        doc_count = graph.query("MATCH (d:Document) RETURN count(d) as count")[0]['count']
        chunk_count = graph.query("MATCH (c:Chunk) RETURN count(c) as count")[0]['count']
        entity_count = graph.query("MATCH (e:Entity) RETURN count(e) as count")[0]['count']
        relationship_count = graph.query("MATCH ()-[r]->() RETURN count(r) as count")[0]['count']
        
        return {
            "documents": doc_count,
            "chunks": chunk_count,
            "entities": entity_count,
            "relationships": relationship_count
        }
    except Exception as e:
        logger.error(f"Error getting graph statistics: {str(e)}", exc_info=True)
        return {
            "documents": 0,
            "chunks": 0,
            "entities": 0,
            "relationships": 0
        }

def reset_neo4j_database(graph: Neo4jGraph) -> None:
    """Reset the Neo4j database by deleting all nodes and relationships
    
    Args:
        graph: Neo4jGraph object with active connection
    """
    logger.info("Resetting Neo4j database - deleting all nodes and relationships")
    try:
        # Delete all nodes and relationships
        query = "MATCH (n) DETACH DELETE n"
        graph.query(query)
        
        # Verify database is empty
        result = graph.query("MATCH (n) RETURN count(n) as count")
        node_count = result[0]['count']
        
        logger.info(f"Database reset complete. Database now has {node_count} nodes")
        
    except Exception as e:
        logger.error(f"Failed to reset Neo4j database: {str(e)}", exc_info=True)
        raise 