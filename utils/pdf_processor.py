import os
import re
import time
from datetime import datetime
from typing import List, Dict, Tuple, Any
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from utils.logging_config import logger

def process_pdfs(pdf_paths: List[str]) -> Tuple[List[Document], Dict[str, Any]]:
    """Process PDFs with error handling, text cleaning, and chunking for graph knowledge base
    
    Args:
        pdf_paths: List of paths to PDF files
        
    Returns:
        Tuple containing:
            - List of document chunks
            - Dictionary of source metadata
    """
    logger.info(f"Starting to process {len(pdf_paths)} PDF files")
    all_chunks = []
    source_metadata = {}

    if not pdf_paths:
        logger.error("No PDF files provided")
        raise ValueError("No PDF files provided")

    for pdf_path in pdf_paths:
        try:
            start_time = time.time()
            logger.info(f"Processing {pdf_path}")

            # Load PDF
            loader = PyMuPDFLoader(pdf_path)
            documents = loader.load()
            logger.debug(f"Loaded {len(documents)} pages from {pdf_path}")

            # Store metadata about the source document
            source_name = os.path.basename(pdf_path)
            source_metadata[source_name] = {
                "filename": source_name,
                "pages": len(documents),
                "processed_at": datetime.now().isoformat()
            }

            # Clean text
            for doc in documents:
                doc.page_content = re.sub(r'\s+', ' ', doc.page_content).strip()
                doc.metadata['source'] = source_name

            # Split into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                separators=["\n\n", "\n", ". ", " ", ""],
                length_function=len
            )

            chunks = text_splitter.split_documents(documents)

            # Add more detailed metadata to chunks
            for i, chunk in enumerate(chunks):
                chunk.metadata['chunk_id'] = f"{source_name}_chunk_{i}"
                chunk.metadata['source_document'] = source_name

            all_chunks.extend(chunks)
            logger.info(f"Processed {pdf_path} into {len(chunks)} chunks in {time.time() - start_time:.2f} seconds")

        except Exception as e:
            logger.error(f"Error processing {pdf_path}: {str(e)}", exc_info=True)
            continue

    logger.info(f"Completed processing all PDFs. Generated {len(all_chunks)} total chunks")
    return all_chunks, source_metadata 