# Graph RAG PDF Analysis App

This Streamlit application provides a user-friendly interface for analyzing PDF documents using a graph-based Retrieval Augmented Generation (RAG) approach. It combines the power of Neo4j graph database, vector embeddings, and the Mistral AI language model.

## Features

- Upload and process multiple PDF documents
- Automatically extract entities and build a knowledge graph
- Ask questions about the content of your documents
- Extract specific properties (materials, connector information, etc.)
- Visualize the knowledge graph structure and relationships

## Setup

1. Clone this repository
2. Install the dependencies: `pip install -r requirements.txt`
3. Create a `.env` file with your Neo4j and Mistral API credentials (see `.env.example`)
4. Run the app: `streamlit run app.py`

## Requirements

- Python 3.8+
- Neo4j database instance
- Mistral AI API key

## Usage

1. Upload your PDF files
2. Wait for the processing to complete
3. Ask questions about your documents or use the property extraction tools 