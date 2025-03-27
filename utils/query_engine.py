import re
import time
import json
from typing import List, Dict, Any
from langchain_community.graphs import Neo4jGraph
from langchain.vectorstores.neo4j_vector import Neo4jVector
import fireworks.client as fw
from utils.logging_config import logger

def extract_entities_from_question(question: str) -> List[str]:
    """Simple keyword extraction for entity matching
    
    Args:
        question: User question text
        
    Returns:
        List of extracted entity keywords
    """
    # Remove common stop words
    stop_words = {'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                  'with', 'by', 'about', 'like', 'through', 'over', 'before', 'after', 
                  'since', 'of', 'from'}

    # Tokenize and filter
    words = re.findall(r'\b\w+\b', question.lower())
    filtered_words = [word for word in words if word not in stop_words and len(word) > 3]

    # Get noun phrases (simple approximation)
    text = question.lower()
    phrases = re.findall(r'\b[a-z]+\s+[a-z]+\b', text)
    phrases = [p for p in phrases if not any(word in stop_words for word in p.split())]

    # Combine single words and phrases
    entities = filtered_words + phrases
    return list(set(entities))  # Remove duplicates

def get_query_context(
    question: str, 
    vector_store: Neo4jVector, 
    graph: Neo4jGraph
) -> List[Dict[str, Any]]:
    """Get context for query using both vector similarity and graph relationships
    
    Args:
        question: User question text
        vector_store: Neo4jVector object for similarity search
        graph: Neo4jGraph object for graph traversal
        
    Returns:
        List of context dictionaries with relevant information
    """
    logger.info(f"Getting context for question: {question}")
    contexts = []

    try:
        # Step 1: Extract potential entities from the question
        start_time = time.time()
        logger.debug("Identifying key entities in the question")
        entities_in_question = extract_entities_from_question(question)
        logger.debug(f"Identified entities: {entities_in_question}")

        # Step 2: Use vector similarity to get relevant chunks
        logger.debug("Performing vector similarity search")
        vector_results = vector_store.similarity_search(question, k=3)
        logger.debug(f"Vector search found {len(vector_results)} relevant chunks")

        for idx, doc in enumerate(vector_results):
            contexts.append({
                "source": "vector_similarity",
                "rank": idx,
                "chunk_id": doc.metadata.get('chunk_id', 'unknown'),
                "content": doc.page_content,
                "source_document": doc.metadata.get('source_document', 'unknown')
            })

        # Step 3: Use graph relationships to find related chunks
        if entities_in_question:
            logger.debug("Performing graph traversal to find related content")
            for entity in entities_in_question:
                # Find chunks that mention this entity or related entities
                query = """
                MATCH (c:Chunk)-[:MENTIONS]->(e:Entity)
                WHERE toLower(e.name) CONTAINS toLower($entity_name)
                RETURN c.id as chunk_id, c.content as content, c.page_num as page_num
                UNION
                MATCH (e1:Entity)-[:RELATED_TO]->(e2:Entity)<-[:MENTIONS]-(c:Chunk)
                WHERE toLower(e1.name) CONTAINS toLower($entity_name)
                RETURN c.id as chunk_id, c.content as content, c.page_num as page_num
                LIMIT 3
                """
                graph_results = graph.query(query, params={"entity_name": entity})

                for idx, result in enumerate(graph_results):
                    contexts.append({
                        "source": "graph_traversal",
                        "entity": entity,
                        "rank": idx,
                        "chunk_id": result.get('chunk_id', 'unknown'),
                        "content": result.get('content', ''),
                        "page_num": result.get('page_num', 0)
                    })

        logger.info(f"Context gathering completed in {time.time() - start_time:.2f} seconds. Found {len(contexts)} relevant contexts")
        return contexts

    except Exception as e:
        logger.error(f"Error getting query context: {str(e)}", exc_info=True)
        return contexts

def answer_question(
    question: str, 
    api_key: str, 
    contexts: List[Dict[str, Any]], 
    neo4j_uri: str, 
    neo4j_username: str, 
    neo4j_password: str
) -> Dict[str, Any]:
    """Generate answer based on retrieved contexts using Fireworks API
    
    Args:
        question: User question text
        api_key: Fireworks API key
        contexts: List of context dictionaries
        neo4j_uri: Neo4j connection URI
        neo4j_username: Neo4j username
        neo4j_password: Neo4j password
        
    Returns:
        Dictionary with answer and metadata
    """
    logger.info(f"Generating answer for question: {question}")
    start_time = time.time()

    try:
        # Compile context
        context_text = "\n\n".join([
            f"Source: {ctx['source']}, ID: {ctx.get('chunk_id', ctx.get('connector_name', 'unknown'))}\nContent: {ctx['content']}"
            for ctx in contexts
        ])

        # Add graph schema information
        graph = Neo4jGraph(url=neo4j_uri, username=neo4j_username, password=neo4j_password)
        schema_text = graph.get_schema

        # Formulating the API prompt
        prompt = f"""You are a specialized document assistant working with a graph-based knowledge system.

        Knowledge Graph Schema:
        {schema_text}

        Context information from documents:
        {context_text}

        Question: {question}

        Answer the question based on the context provided. If you're unsure or the information isn't in the context, say "I don't have enough information in my knowledge base to answer this question properly."

        Include references to document sources when possible. Be precise and concise."""

        # Initialize Fireworks API key - FIXED: Using correct API key attribute
        fw.api_key = api_key

        # Send request to Fireworks API
        response = fw.ChatCompletion.create(
            model="accounts/fireworks/models/llama-v3p1-8b-instruct",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
            temperature=0.3
        )

        # Handle Fireworks response format
        if response.choices and len(response.choices) > 0:
            message_content = response.choices[0].message.content
        else:
            raise ValueError("No response content found in Fireworks API response.")

        # Clean up the response
        cleaned_answer = re.sub(r'\\boxed{', '', message_content)  # Remove LaTeX box start
        cleaned_answer = re.sub(r'\\[^\s]+', '', cleaned_answer)   # Remove other LaTeX commands
        cleaned_answer = cleaned_answer.replace('\\n', '\n')       # Convert escaped newlines

        # Format the result
        result = {
            "answer": cleaned_answer.strip(),
            "processing_time": time.time() - start_time,
            "contexts_used": len(contexts),
            "context_sources": list(set([c['source'] for c in contexts]))
        }

        logger.info(f"Answer generated in {time.time() - start_time:.2f} seconds")
        return result

    except Exception as e:
        logger.error(f"Error generating answer: {str(e)}", exc_info=True)
        return {"error": f"Error generating answer: {str(e)}"} 