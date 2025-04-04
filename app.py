import streamlit as st
import os
import time
from typing import Dict, Any, List, Optional
import tempfile
import concurrent.futures
from dotenv import load_dotenv

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

# Import components
from components.results import (
    display_processing_status,
    display_file_info,
    display_graph_stats
)

# Import prompts
from prompts.extraction_prompts import (
    # Material Properties
    MATERIAL_PROMPT,
    MATERIAL_NAME_PROMPT,

    # Physical / Mechanical Attributes
    PULL_TO_SEAT_PROMPT,
    GENDER_PROMPT,
    HEIGHT_MM_PROMPT,
    LENGTH_MM_PROMPT,
    WIDTH_MM_PROMPT,
    NUMBER_OF_CAVITIES_PROMPT,
    NUMBER_OF_ROWS_PROMPT,
    MECHANICAL_CODING_PROMPT,
    COLOUR_PROMPT,
    COLOUR_CODING_PROMPT,

    # Sealing & Environmental
    WORKING_TEMPERATURE_PROMPT,
    HOUSING_SEAL_PROMPT,
    WIRE_SEAL_PROMPT,
    SEALING_PROMPT,
    SEALING_CLASS_PROMPT,

    # Terminals & Connections
    CONTACT_SYSTEMS_PROMPT,
    TERMINAL_POSITION_ASSURANCE_PROMPT,
    CONNECTOR_POSITION_ASSURANCE_PROMPT,
    CLOSED_CAVITIES_PROMPT,

    # Assembly & Type
    PRE_ASSEMBLED_PROMPT,
    CONNECTOR_TYPE_PROMPT,
    SET_KIT_PROMPT,

    # Specialized Attributes
    HV_QUALIFIED_PROMPT
)

# Initialize dotenv for environment variables
load_dotenv()

# Initialize session state if not already done
if 'graph_data' not in st.session_state:
    st.session_state.graph_data = None

if 'extraction_results' not in st.session_state:
    st.session_state.extraction_results = {}

if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False

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
        api_key: Fireworks API key
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
        api_key: Fireworks API key
        graph_data: Graph data dictionary
        
    Returns:
        Answer result dictionary
    """
    try:
        if not graph_data or "graph" not in graph_data:
            raise ValueError("Graph data not initialized. Process PDFs first.")

        if not api_key:
            raise ValueError("Missing DeepSeek API key")

        # Display processing status
        with st.spinner("Retrieving context and generating answer..."):
            # Get combined context from vector and graph
            contexts = get_query_context(
                question,
                graph_data["vector_store"],
                graph_data["graph"].query,
                api_key,
                k_vector=3,
                k_graph=5,
                use_llm_extraction=True
            )

            # Get answer using context
            result = answer_question(
                question,
                api_key,
                contexts
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
    """Run an extraction based on the selected option with enhanced graph traversal
    
    Args:
        extraction_option: Type of extraction to run
        api_key: Fireworks API key
        graph_data: Graph data dictionary
        
    Returns:
        Extraction result dictionary
    """
    try:
        if not graph_data or "graph" not in graph_data:
            raise ValueError("Graph data not initialized. Process PDFs first.")

        if not api_key:
            raise ValueError("Missing Fireworks API key")

        # Select the appropriate prompt based on the extraction option
        if extraction_option == "Material Filling":
            prompt = MATERIAL_PROMPT
        elif extraction_option == "Material Name":
            prompt = MATERIAL_NAME_PROMPT
        elif extraction_option == "Connector Gender":
            prompt = GENDER_PROMPT
        elif extraction_option == "Pull To Seat":
            prompt = PULL_TO_SEAT_PROMPT
        elif extraction_option == "Height (mm)":
            prompt = HEIGHT_MM_PROMPT
        elif extraction_option == "Length (mm)":
            prompt = LENGTH_MM_PROMPT
        elif extraction_option == "Width (mm)":
            prompt = WIDTH_MM_PROMPT
        elif extraction_option == "Number of Cavities":
            prompt = NUMBER_OF_CAVITIES_PROMPT
        elif extraction_option == "Number of Rows":
            prompt = NUMBER_OF_ROWS_PROMPT
        elif extraction_option == "Mechanical Coding":
            prompt = MECHANICAL_CODING_PROMPT
        elif extraction_option == "Colour":
            prompt = COLOUR_PROMPT
        elif extraction_option == "Colour Coding":
            prompt = COLOUR_CODING_PROMPT
        elif extraction_option == "Working Temperature":
            prompt = WORKING_TEMPERATURE_PROMPT
        elif extraction_option == "Housing Seal":
            prompt = HOUSING_SEAL_PROMPT
        elif extraction_option == "Wire Seal":
            prompt = WIRE_SEAL_PROMPT
        elif extraction_option == "Sealing":
            prompt = SEALING_PROMPT
        elif extraction_option == "Sealing Class":
            prompt = SEALING_CLASS_PROMPT
        elif extraction_option == "Contact Systems":
            prompt = CONTACT_SYSTEMS_PROMPT
        elif extraction_option == "Terminal Position Assurance":
            prompt = TERMINAL_POSITION_ASSURANCE_PROMPT
        elif extraction_option == "Connector Position Assurance":
            prompt = CONNECTOR_POSITION_ASSURANCE_PROMPT
        elif extraction_option == "Closed Cavities":
            prompt = CLOSED_CAVITIES_PROMPT
        elif extraction_option == "Pre-Assembled":
            prompt = PRE_ASSEMBLED_PROMPT
        elif extraction_option == "Connector Type":
            prompt = CONNECTOR_TYPE_PROMPT
        elif extraction_option == "Set Kit":
            prompt = SET_KIT_PROMPT
        elif extraction_option == "HV Qualified":
            prompt = HV_QUALIFIED_PROMPT
        else:
            raise ValueError(f"Unknown extraction option: {extraction_option}")
        
        # Enhanced extraction with specific graph traversal instructions
        logger.info(f"Running extraction for: {extraction_option}")
        
        # Add extraction-specific instructions to the prompt
        augmented_prompt = f"""EXTRACTION TYPE: {extraction_option}

{prompt}

IMPORTANT: Use the graph database to find connections between entities for this extraction.
"""
        
        # Use the enhanced Q&A function with the extraction prompt
        result = ask_graph_rag(augmented_prompt, api_key, graph_data)
        return result

    except Exception as e:
        logger.error(f"Error in extraction pipeline: {str(e)}", exc_info=True)
        return {"error": f"Error running extraction: {str(e)}"}

def run_all_extractions(api_key: str, graph_data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Run all extraction types in parallel
    
    Args:
        api_key: Fireworks API key
        graph_data: Graph data dictionary
        
    Returns:
        Dictionary of extraction results by type
    """
    extraction_types = [
        "Material Filling", 
        "Material Name", 
        "Connector Gender",
        "Pull To Seat",
        "Height (mm)",
        "Length (mm)",
        "Width (mm)",
        "Number of Cavities",
        "Number of Rows",
        "Mechanical Coding",
        "Colour",
        "Colour Coding",
        "Working Temperature",
        "Housing Seal",
        "Wire Seal",
        "Sealing",
        "Sealing Class",
        "Contact Systems",
        "Terminal Position Assurance",
        "Connector Position Assurance",
        "Closed Cavities",
        "Pre-Assembled",
        "Connector Type",
        "Set Kit",
        "HV Qualified"
    ]
    
    results = {}
    
    with st.status("Running extractions...", expanded=True) as status:
        # Use ThreadPoolExecutor to run extractions in parallel
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Create a future for each extraction type
            future_to_extraction = {
                executor.submit(run_extraction, extraction_type, api_key, graph_data): extraction_type
                for extraction_type in extraction_types
            }
            
            # Process results as they complete
            for i, future in enumerate(concurrent.futures.as_completed(future_to_extraction)):
                extraction_type = future_to_extraction[future]
                st.write(f"Completed {extraction_type} extraction ({i+1}/{len(extraction_types)})")
                try:
                    result = future.result()
                    results[extraction_type] = result
                except Exception as e:
                    logger.error(f"Error in {extraction_type} extraction: {str(e)}", exc_info=True)
                    results[extraction_type] = {"error": str(e)}
        
        status.update(label="All extractions complete!", state="complete")
    
    return results

def display_extraction_table(extraction_results: Dict[str, Dict[str, Any]]):
    """Display all extraction results in a table format
    
    Args:
        extraction_results: Dictionary of extraction results by type
    """
    st.subheader("📊 Extraction Results")
    
    if not extraction_results:
        st.info("No extraction results available")
        return
    
    # Create a table for the results
    table_data = []
    
    for extraction_type, result in extraction_results.items():
        if "error" in result:
            table_data.append({
                "Extraction Type": extraction_type,
                "Status": "❌ Error",
                "Result": result["error"]
            })
        else:
            # Get the answer from the result
            answer = result.get("answer", "No result")
            table_data.append({
                "Extraction Type": extraction_type,
                "Status": "✅ Success",
                "Result": answer
            })
    
    # Display as DataFrame
    import pandas as pd
    df = pd.DataFrame(table_data)
    st.dataframe(df, use_container_width=True)

def main():
    """Main application entry point"""
    # Set page configuration
    st.set_page_config(
        page_title="Document Analysis Tool",
        page_icon="📄",
        layout="wide"
    )
    
    # Get configuration from environment variables or secrets
    neo4j_uri = os.environ.get("NEO4J_URI", "neo4j://localhost:7687")
    neo4j_username = os.environ.get("NEO4J_USERNAME", "neo4j")
    neo4j_password = os.environ.get("NEO4J_PASSWORD", "")
    deepseek_api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    
    # Use Streamlit secrets for sensitive info if available
    if hasattr(st, "secrets"):
        if "neo4j_password" in st.secrets:
            neo4j_password = st.secrets["neo4j_password"]
        if "deepseek_api_key" in st.secrets:
            deepseek_api_key = st.secrets["deepseek_api_key"]
    
    # Always reset the database
    reset_db = True
    
    # Main content area
    st.title("📄 Document Analysis Tool")
    
    # File upload section - now in the main content area
    pdf_files = st.file_uploader(
        "Upload PDF Documents", 
        type="pdf", 
        accept_multiple_files=True,
        help="Upload one or more PDF files for analysis"
    )
    
    # Save uploaded files to temp directory
    pdf_paths = []
    if pdf_files:
        temp_dir = tempfile.mkdtemp()
        for pdf_file in pdf_files:
            temp_path = os.path.join(temp_dir, pdf_file.name)
            with open(temp_path, "wb") as f:
                f.write(pdf_file.getvalue())
            pdf_paths.append(temp_path)
    
    # Process button - now in the main content area
    process_button = st.button("Process Documents", type="primary", disabled=not pdf_files)
    
    # Initial state instructions
    if not st.session_state.processing_complete and not process_button:
        if not pdf_files:
            st.info("Start by uploading your documents above")
            st.write("This tool will:")
            st.write("1. Process your PDF documents")
            st.write("2. Build a knowledge graph")
            st.write("3. Automatically extract key information")
            st.write("4. Display the results in a table")
            
            # Show example image or placeholder
            st.image("https://via.placeholder.com/800x400?text=Upload+Documents+to+Start", use_container_width=True)
    
    # Process files if button clicked
    if process_button:
        if not pdf_paths:
            st.warning("⚠️ Please upload PDF files first")
        else:
            # Process the files and build graph
            with st.container():
                st.subheader("🔄 Processing Documents")
                st.session_state.graph_data = process_files_and_build_graph(
                    pdf_paths,
                    neo4j_uri,
                    neo4j_username,
                    neo4j_password,
                    deepseek_api_key,
                    reset_db
                )
                
                if st.session_state.graph_data:
                    # Run all extractions automatically
                    st.subheader("🔍 Running Information Extraction")
                    st.session_state.extraction_results = run_all_extractions(
                        deepseek_api_key,
                        st.session_state.graph_data
                    )
                    st.session_state.processing_complete = True
    
    # Show file information if files are uploaded
    if pdf_paths:
        with st.expander("📁 Uploaded Documents", expanded=True):
            display_file_info(pdf_paths)
    
    # Display extraction results if available
    if st.session_state.processing_complete and st.session_state.extraction_results:
        display_extraction_table(st.session_state.extraction_results)

if __name__ == "__main__":
    main() 