import streamlit as st
import os
from dotenv import load_dotenv
from typing import Dict, Any, Tuple, List
import tempfile

from utils.logging_config import logger

def load_env_variables() -> Dict[str, str]:
    """Load environment variables from .env file"""
    load_dotenv()
    
    env_vars = {
        "neo4j_uri": os.getenv("NEO4J_URI", ""),
        "neo4j_username": os.getenv("NEO4J_USERNAME", ""),
        "neo4j_password": os.getenv("NEO4J_PASSWORD", ""),
        "mistral_api_key": os.getenv("MISTRAL_API_KEY", "")
    }
    
    return env_vars

def create_sidebar() -> Dict[str, Any]:
    """Create and display the sidebar with configuration options
    
    Returns:
        Dictionary with all sidebar configurations
    """
    st.sidebar.title("📊 Graph RAG App")
    
    # Load environment variables
    env_vars = load_env_variables()
    
    # Configuration section
    st.sidebar.header("⚙️ Configuration")
    
    # Neo4j configuration
    with st.sidebar.expander("Neo4j Connection", expanded=False):
        neo4j_uri = st.text_input(
            "Neo4j URI", 
            value=env_vars["neo4j_uri"],
            placeholder="bolt://localhost:7687"
        )
        
        neo4j_username = st.text_input(
            "Neo4j Username", 
            value=env_vars["neo4j_username"],
            placeholder="neo4j"
        )
        
        neo4j_password = st.text_input(
            "Neo4j Password", 
            value=env_vars["neo4j_password"],
            placeholder="password",
            type="password"
        )
    
    # Mistral API key
    with st.sidebar.expander("Mistral AI API", expanded=False):
        mistral_api_key = st.text_input(
            "Mistral API Key", 
            value=env_vars["mistral_api_key"],
            placeholder="Enter your API key",
            type="password"
        )
    
    # File upload section
    st.sidebar.header("📁 Document Upload")
    
    uploaded_files = st.sidebar.file_uploader(
        "Upload PDF Documents", 
        type=["pdf"], 
        accept_multiple_files=True
    )
    
    # Process PDFs button
    process_button = st.sidebar.button("Process Documents", type="primary", use_container_width=True)
    
    # Save uploaded files to temp directory for processing
    pdf_paths = []
    if uploaded_files:
        # Create a temporary directory to store the uploaded files
        temp_dir = tempfile.mkdtemp()
        
        for uploaded_file in uploaded_files:
            if uploaded_file.name.lower().endswith('.pdf'):
                file_path = os.path.join(temp_dir, uploaded_file.name)
                
                # Save uploaded file to temporary location
                with open(file_path, 'wb') as f:
                    f.write(uploaded_file.getvalue())
                    
                pdf_paths.append(file_path)
                logger.info(f"Saved uploaded file: {file_path}")
    
    # Analysis Tools
    st.sidebar.header("🔍 Analysis Tools")
    
    # Add a radio button for the different extraction options
    extraction_option = st.sidebar.radio(
        "Extraction Type",
        ["Material Filling", "Material Name", "Connector Gender"],
        index=0
    )
    
    run_extraction = st.sidebar.button(
        "Run Extraction", 
        type="secondary", 
        use_container_width=True
    )
    
    # About section
    st.sidebar.header("ℹ️ About")
    st.sidebar.info(
        """
        This application uses a graph-based RAG approach to analyze PDF documents.
        
        - Upload your PDFs
        - Process them to build a knowledge graph
        - Ask questions or run extractions
        
        Built with Streamlit, Neo4j, and Mistral AI.
        """
    )
    
    # Return all the sidebar configurations
    return {
        "neo4j_uri": neo4j_uri,
        "neo4j_username": neo4j_username,
        "neo4j_password": neo4j_password,
        "mistral_api_key": mistral_api_key,
        "pdf_paths": pdf_paths,
        "process_button": process_button,
        "extraction_option": extraction_option,
        "run_extraction": run_extraction
    } 