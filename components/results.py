import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import time
from typing import Dict, Any, List, Optional
import os

def display_processing_status(status: str, progress: float = None):
    """Display the current processing status with progress bar
    
    Args:
        status: Status message to display
        progress: Optional progress value (0-1)
    """
    if progress is not None:
        progress_bar = st.progress(progress)
        st.caption(status)
    else:
        with st.spinner(status):
            time.sleep(0.1)  # Small delay to ensure spinner is shown

def display_file_info(pdf_paths: List[str]):
    """Display information about uploaded files
    
    Args:
        pdf_paths: List of PDF file paths
    """
    if not pdf_paths:
        st.info("No files uploaded yet. Please upload PDF files using the sidebar.")
        return
    
    st.subheader("📄 Uploaded Documents")
    
    file_data = []
    for path in pdf_paths:
        filename = path.split("/")[-1]
        file_data.append({
            "Filename": filename,
            "Size": f"{os.path.getsize(path) / 1024:.1f} KB"
        })
    
    file_df = pd.DataFrame(file_data)
    st.dataframe(file_df, use_container_width=True)

def display_graph_stats(stats: Dict[str, int]):
    """Display knowledge graph statistics with visualizations
    
    Args:
        stats: Dictionary with graph statistics
    """
    st.subheader("📊 Knowledge Graph Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Documents", stats["documents"])
    
    with col2:
        st.metric("Chunks", stats["chunks"])
    
    with col3:
        st.metric("Entities", stats["entities"])
    
    with col4:
        st.metric("Relationships", stats["relationships"])
    
    # Create a simple bar chart of the statistics
    fig = px.bar(
        x=list(stats.keys()),
        y=list(stats.values()),
        labels={'x': 'Component', 'y': 'Count'},
        title='Knowledge Graph Components',
        color=list(stats.keys()),
        color_discrete_sequence=px.colors.qualitative.Plotly
    )
    
    st.plotly_chart(fig, use_container_width=True)

def display_answer(result: Dict[str, Any], question: str):
    """Display the formatted answer and metadata
    
    Args:
        result: Answer result dictionary
        question: Original question
    """
    if "error" in result:
        st.error(f"Error: {result['error']}")
        return
    
    st.subheader("❓ Question")
    st.write(question)
    
    st.subheader("💡 Answer")
    st.markdown(result["answer"])
    
    # Display metadata in an expander
    with st.expander("Answer Metadata"):
        st.write(f"⏱️ Processing time: {result['processing_time']:.2f} seconds")
        st.write(f"📑 Context chunks used: {result['contexts_used']}")
        st.write(f"🔍 Context sources: {', '.join(result['context_sources'])}")

def display_extraction_results(result: Dict[str, Any], extraction_type: str):
    """Display extraction results with proper formatting
    
    Args:
        result: Extraction result dictionary
        extraction_type: Type of extraction performed
    """
    if "error" in result:
        st.error(f"Error: {result['error']}")
        return
    
    st.subheader(f"✨ {extraction_type} Extraction Results")
    
    # Display the raw answer
    st.markdown(result["answer"])
    
    # Try to extract structured data if possible
    try:
        # Look for lists or tables in the output
        if "- " in result["answer"] or "* " in result["answer"]:
            st.subheader("Structured Data")
            
            # Simple parsing of list items
            lines = result["answer"].split("\n")
            items = [line.strip("- *").strip() for line in lines if line.strip().startswith(("- ", "* "))]
            
            if items:
                for item in items:
                    st.write(f"• {item}")
    except Exception as e:
        # If parsing fails, just show the raw result
        pass
    
    # Display metadata in an expander
    with st.expander("Extraction Metadata"):
        st.write(f"⏱️ Processing time: {result['processing_time']:.2f} seconds")
        st.write(f"📑 Context chunks used: {result['contexts_used']}")
        st.write(f"🔍 Context sources: {', '.join(result['context_sources'])}")

def display_visualization(graph_data: Dict[str, Any]):
    """Display graph visualization
    
    Args:
        graph_data: Graph data for visualization
    """
    st.subheader("🔍 Knowledge Graph Visualization")
    
    if not graph_data or "graph" not in graph_data:
        st.info("Process documents first to visualize the knowledge graph")
        return
    
    # Get entity-relationship data for visualization
    try:
        graph = graph_data["graph"]
        
        # Query to get top entities by connections
        entity_query = """
        MATCH (e:Entity)
        WITH e, size((e)-[:RELATED_TO]-()) as connections
        RETURN e.name as entity, e.type as type, connections
        ORDER BY connections DESC
        LIMIT 20
        """
        
        entities = graph.query(entity_query)
        
        if not entities:
            st.info("No entity data available for visualization")
            return
        
        # Create entity dataframe
        entity_df = pd.DataFrame(entities)
        
        # Create a bubble chart
        fig = px.scatter(
            entity_df, 
            x="connections", 
            y="type",
            size="connections",
            color="type",
            hover_name="entity",
            text="entity",
            title="Top Entities by Connection Count",
            size_max=50
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Query to get relationships between top entities
        rel_query = """
        MATCH (e1:Entity)-[r:RELATED_TO]->(e2:Entity)
        WITH e1, e2, r
        ORDER BY r.weight DESC
        LIMIT 50
        RETURN e1.name as source, e2.name as target, r.weight as weight
        """
        
        relationships = graph.query(rel_query)
        
        if relationships:
            st.subheader("Entity Relationships")
            rel_df = pd.DataFrame(relationships)
            st.dataframe(rel_df, use_container_width=True)
            
            # More advanced network graph could be added here
            # But it's more complex and may require custom JavaScript
            
    except Exception as e:
        st.error(f"Error generating visualization: {str(e)}") 