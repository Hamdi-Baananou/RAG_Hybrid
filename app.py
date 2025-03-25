import streamlit as st
import os
import time
from typing import Dict, Any, List, Optional
import tempfile

# Import utility modules
from utils.logging_config import logger
from utils.embeddings import get_embeddings
from utils.pdf_processor import process_pdfs
from utils.graph_db import (
    connect_to_neo4j, 
    setup_neo4j_schema, 
    create_knowledge_graph, 
    setup_vector_index,
    get_graph_statistics,
    reset_neo4j_database
)
from utils.query_engine import (
    get_query_context,
    answer_question
)

# Import prompts
from prompts.extraction_prompts import (
    MATERIAL_PROMPT,
    MATERIAL_NAME_PROMPT,
    GENDER_PROMPT
)

# Import components
from components.sidebar import create_sidebar
from components.results import (
    display_processing_status,
    display_file_info,
    display_graph_stats,
    display_answer,
    display_extraction_results,
    display_visualization
)

# Initialize session state if not already done
if 'graph_data' not in st.session_state:
    st.session_state.graph_data = None

if 'last_question' not in st.session_state:
    st.session_state.last_question = None

if 'last_answer' not in st.session_state:
    st.session_state.last_answer = None

if 'last_extraction' not in st.session_state:
    st.session_state.last_extraction = None

if 'last_extraction_result' not in st.session_state:
    st.session_state.last_extraction_result = None

def process_files_and_build_graph(
    pdf_paths: List[str], 
    neo4j_uri: str, 
    neo4j_username: str, 
    neo4j_password: str, 
    api_key: str,
    reset_db: bool = False
) -> Dict[str, Any]:
    """Process PDF files and build the knowledge graph
    
    Args:
        pdf_paths: List of PDF file paths
        neo4j_uri: Neo4j connection URI
        neo4j_username: Neo4j username
        neo4j_password: Neo4j password
        api_key: Mistral API key
        reset_db: Whether to reset the database before processing
        
    Returns:
        Dictionary with graph data and metadata
    """
    try:
        # Step 1: Process PDFs
        display_processing_status("Processing PDF files...", 0.1)
        chunks, source_metadata = process_pdfs(pdf_paths)
        
        if not chunks:
            st.error("No valid chunks extracted from PDFs")
            return None

        # Step 2: Connect to Neo4j
        display_processing_status("Connecting to Neo4j and setting up schema...", 0.2)
        graph = connect_to_neo4j(neo4j_uri, neo4j_username, neo4j_password)
        
        # Reset database if option is selected
        if reset_db:
            display_processing_status("Resetting Neo4j database...", 0.25)
            reset_neo4j_database(graph)
            
        setup_neo4j_schema(graph)

        # Step 3: Initialize embedding model
        display_processing_status("Initializing embedding model...", 0.3)
        embedding_function = get_embeddings()

        # Step 4: Create knowledge graph
        display_processing_status("Building knowledge graph from document chunks...", 0.5)
        create_knowledge_graph(graph, chunks, source_metadata, api_key)

        # Step 5: Setup vector index
        display_processing_status("Creating vector index for semantic search...", 0.8)
        vector_store = setup_vector_index(chunks, embedding_function, neo4j_uri, neo4j_username, neo4j_password)

        # Step 6: Generate graph statistics
        display_processing_status("Generating graph statistics...", 0.9)
        stats = get_graph_statistics(graph)

        # Store everything needed for retrieval
        result_data = {
            "graph": graph,
            "vector_store": vector_store,
            "neo4j_uri": neo4j_uri,
            "neo4j_username": neo4j_username,
            "neo4j_password": neo4j_password,
            "statistics": stats
        }

        display_processing_status("Knowledge graph built successfully!", 1.0)
        return result_data

    except Exception as e:
        logger.error(f"Critical error in processing pipeline: {str(e)}", exc_info=True)
        st.error(f"Error processing files: {str(e)}")
        return None

def ask_graph_rag(
    question: str, 
    api_key: str, 
    graph_data: Dict[str, Any]
) -> Dict[str, Any]:
    """Ask a question using the graph RAG system
    
    Args:
        question: User question
        api_key: Mistral API key
        graph_data: Graph data dictionary
        
    Returns:
        Answer result dictionary
    """
    try:
        if not graph_data or "graph" not in graph_data:
            raise ValueError("Graph data not initialized. Process PDFs first.")

        if not api_key:
            raise ValueError("Missing Mistral API key")

        # Display processing status
        with st.spinner("Retrieving context and generating answer..."):
            # Get combined context from vector and graph
            contexts = get_query_context(
                question,
                graph_data["vector_store"],
                graph_data["graph"]
            )

            # Get answer using context
            result = answer_question(
                question,
                api_key,
                contexts,
                graph_data["neo4j_uri"],
                graph_data["neo4j_username"],
                graph_data["neo4j_password"]
            )

            return result

    except Exception as e:
        logger.error(f"Error in QA pipeline: {str(e)}", exc_info=True)
        return {"error": f"Error processing question: {str(e)}"}

def run_extraction(
    extraction_option: str,
    api_key: str,
    graph_data: Dict[str, Any]
) -> Dict[str, Any]:
    """Run an extraction based on the selected option
    
    Args:
        extraction_option: Type of extraction to run
        api_key: Mistral API key
        graph_data: Graph data dictionary
        
    Returns:
        Extraction result dictionary
    """
    try:
        if not graph_data or "graph" not in graph_data:
            raise ValueError("Graph data not initialized. Process PDFs first.")

        if not api_key:
            raise ValueError("Missing Mistral API key")

        # Select the appropriate prompt based on the extraction option
        if extraction_option == "Material Filling":
            prompt = MATERIAL_PROMPT
        elif extraction_option == "Material Name":
            prompt = MATERIAL_NAME_PROMPT
        elif extraction_option == "Connector Gender":
            prompt = GENDER_PROMPT
        else:
            raise ValueError(f"Unknown extraction option: {extraction_option}")

        # Display processing status
        with st.spinner(f"Running {extraction_option} extraction..."):
            # Use the same Q&A function but with the extraction prompt
            result = ask_graph_rag(prompt, api_key, graph_data)
            return result

    except Exception as e:
        logger.error(f"Error in extraction pipeline: {str(e)}", exc_info=True)
        return {"error": f"Error running extraction: {str(e)}"}

def main():
    """Main application entry point"""
    # Set page configuration
    st.set_page_config(
        page_title="Graph RAG PDF Analysis",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Create sidebar and get configurations
    sidebar_config = create_sidebar()

    # Main content area
    st.title("📊 Graph RAG PDF Analysis")
    
    # Process files if button clicked
    if sidebar_config["process_button"]:
        if not sidebar_config["pdf_paths"]:
            st.warning("Please upload PDF files first")
        elif not sidebar_config["neo4j_uri"] or not sidebar_config["neo4j_username"] or not sidebar_config["neo4j_password"]:
            st.warning("Please provide Neo4j connection details")
        elif not sidebar_config["mistral_api_key"]:
            st.warning("Please provide a Mistral API key")
        else:
            # Process the files and build graph
            st.session_state.graph_data = process_files_and_build_graph(
                sidebar_config["pdf_paths"],
                sidebar_config["neo4j_uri"],
                sidebar_config["neo4j_username"],
                sidebar_config["neo4j_password"],
                sidebar_config["mistral_api_key"],
                sidebar_config["reset_db"]
            )
    
    # Show file information
    display_file_info(sidebar_config["pdf_paths"])
    
    # Show graph statistics if available
    if st.session_state.graph_data:
        display_graph_stats(st.session_state.graph_data["statistics"])
    
    # Create two tabs for Q&A and Visualization
    tab1, tab2 = st.tabs(["Question Answering", "Graph Visualization"])
    
    with tab1:
        st.subheader("🔎 Ask Questions About Your Documents")
        
        # Text input for questions
        question = st.text_input("Enter your question:")
        
        col1, col2 = st.columns([1, 4])
        
        with col1:
            ask_button = st.button("Ask Question", type="primary", use_container_width=True)
        
        with col2:
            if ask_button and question:
                if not st.session_state.graph_data:
                    st.warning("Please process PDF files first")
                elif not sidebar_config["mistral_api_key"]:
                    st.warning("Please provide a Mistral API key")
                else:
                    # Send question to the graph RAG system
                    st.session_state.last_question = question
                    st.session_state.last_answer = ask_graph_rag(
                        question,
                        sidebar_config["mistral_api_key"],
                        st.session_state.graph_data
                    )
        
        # Display the last answer if available
        if st.session_state.last_question and st.session_state.last_answer:
            display_answer(st.session_state.last_answer, st.session_state.last_question)
        
        # Run extraction if button clicked
        if sidebar_config["run_extraction"]:
            if not st.session_state.graph_data:
                st.warning("Please process PDF files first")
            elif not sidebar_config["mistral_api_key"]:
                st.warning("Please provide a Mistral API key")
            else:
                st.session_state.last_extraction = sidebar_config["extraction_option"]
                st.session_state.last_extraction_result = run_extraction(
                    sidebar_config["extraction_option"],
                    sidebar_config["mistral_api_key"],
                    st.session_state.graph_data
                )
        
        # Display the last extraction result if available
        if st.session_state.last_extraction and st.session_state.last_extraction_result:
            display_extraction_results(
                st.session_state.last_extraction_result,
                st.session_state.last_extraction
            )
    
    with tab2:
        if st.session_state.graph_data:
            display_visualization(st.session_state.graph_data)
        else:
            st.info("Process documents first to see the knowledge graph visualization")

if __name__ == "__main__":
    main() 