"""
Document retriever implementation for the AI Research Agent.
This module handles retrieving documents from various sources and integrating with the knowledge base.
"""

import os
from typing import List, Dict, Any, Optional
import tempfile
import requests
from langchain_core.documents import Document
from langchain.tools.retriever import create_retriever_tool
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
    UnstructuredMarkdownLoader,
    CSVLoader,
    WebBaseLoader,
    RecursiveUrlLoader,
    PlaywrightURLLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter

from app.core.knowledge_base import KnowledgeBase


class DocumentRetriever:
    """Document retriever for loading and retrieving documents from various sources."""

    def __init__(self, knowledge_base: KnowledgeBase):
        """
        Initialize the Document Retriever.

        Args:
            knowledge_base: KnowledgeBase instance for document storage and retrieval
        """
        self.knowledge_base = knowledge_base
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
    def get_retriever_tool(self):
        """
        Create a retriever tool for the agent.
        
        Returns:
            Retriever tool for the agent
        """
        retriever = self.knowledge_base.get_retriever()
        return create_retriever_tool(
            retriever,
            "knowledge_base_search",
            "Search for information in the knowledge base. Use this tool when you need to access information from documents that have been added to your knowledge base."
        )
        
    def add_document(self, document_path: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Add a document to the knowledge base.
        
        Args:
            document_path: Path to the document
            metadata: Optional metadata about the document
            
        Returns:
            Status message
        """
        if not os.path.exists(document_path):
            return f"Error: Document {document_path} does not exist"
            
        try:
            # Load the document based on its file extension
            file_extension = os.path.splitext(document_path)[1].lower()
            
            if file_extension == ".pdf":
                loader = PyPDFLoader(document_path)
            elif file_extension == ".docx":
                loader = Docx2txtLoader(document_path)
            elif file_extension == ".md" or file_extension == ".markdown":
                loader = UnstructuredMarkdownLoader(document_path)
            elif file_extension == ".csv":
                loader = CSVLoader(document_path)
            elif file_extension == ".txt":
                loader = TextLoader(document_path)
            else:
                return f"Error: Unsupported file extension {file_extension}"
                
            # Load the document
            documents = loader.load()
            
            # Add metadata if provided
            if metadata:
                for doc in documents:
                    doc.metadata.update(metadata)
                    
            # Add source and timestamp to metadata
            for doc in documents:
                doc.metadata["source"] = document_path
                doc.metadata["timestamp"] = str(os.path.getmtime(document_path))
                
            # Split the documents
            split_docs = self.text_splitter.split_documents(documents)
            
            # Add the documents to the knowledge base
            result = self.knowledge_base.add_documents(split_docs)
            
            return f"Successfully processed {document_path}: {result}"
        except Exception as e:
            return f"Error processing {document_path}: {str(e)}"
            
    def add_url_content(self, url: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Add content from a URL to the knowledge base.
        
        Args:
            url: URL to fetch and learn from
            metadata: Optional metadata about the content
                - max_depth: Optional[int] - maximum recursion depth for fetching linked URLs
            
        Returns:
            Status message
        """
        import logging
        import hashlib
        from urllib.parse import urlparse, urljoin
        from bs4 import BeautifulSoup
        import requests
    
        logger = logging.getLogger(__name__)
        
        # Extract max_depth from metadata if provided
        max_depth = 0
        if metadata and "max_depth" in metadata:
            max_depth = int(metadata["max_depth"])
            # Remove max_depth from metadata as we don't want to store it
            metadata_copy = metadata.copy()
            metadata_copy.pop("max_depth")
            metadata = metadata_copy

        # Create a set to track processed URLs to avoid duplicates
        processed_urls = set()
        results = []
    
        def process_url_recursive(current_url, current_depth=0, max_depth=0):
            """Process URL and recursively process linked URLs up to max_depth"""
            
            # Skip if already processed
            if current_url in processed_urls:
                logger.info(f"Skipping already processed URL: {current_url}")
                return 0
            
            # Add to processed URLs
            processed_urls.add(current_url)
            
            # Log progress with depth indication
            depth_indicator = "  " * current_depth
            logger.info(f"{depth_indicator}Processing URL: {current_url} (depth {current_depth}/{max_depth})")
            
            try:
                # Get the URL content
                logger.info(f"{depth_indicator}Fetching content from {current_url}")
                loader = WebBaseLoader(current_url)
                documents = loader.load()
                
                # Calculate content hash for deduplication
                content = documents[0].page_content if documents else ""
                content_length = len(content)
                content_hash = hashlib.md5(content.encode()).hexdigest()
                logger.info(f"{depth_indicator}Content length: {content_length} characters, hash: {content_hash[:8]}...")
                
                # Set metadata for documents
                url_meta = {}
                if metadata:
                    url_meta.update(metadata)
                
                # Always set document_id to URL for deduplication
                url_meta["document_id"] = current_url
                url_meta["content_hash"] = content_hash
                url_meta["content_length"] = content_length
                
                # Add metadata to documents
                for doc in documents:
                    doc.metadata.update(url_meta)
                
                # Process the documents
                logger.info(f"{depth_indicator}Splitting content into chunks...")
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
                )
                split_docs = text_splitter.split_documents(documents)
                logger.info(f"{depth_indicator}Created {len(split_docs)} document chunks")
                
                # Add to knowledge base
                logger.info(f"{depth_indicator}Adding documents to knowledge base...")
                result = self.knowledge_base.add_documents(split_docs)
                logger.info(f"{depth_indicator}Knowledge base update complete: {result}")
                
                # Don't process links if we've reached max depth
                processed_count = 1  # Count this URL
                if current_depth < max_depth:
                    # Extract links for further processing
                    try:
                        # Parse the URL to get domain
                        parsed_url = urlparse(current_url)
                        base_domain = parsed_url.netloc
                        
                        logger.info(f"{depth_indicator}Extracting links from {current_url}...")
                        # Get HTML content with requests
                        response = requests.get(current_url)
                        soup = BeautifulSoup(response.content, 'html.parser')
                        
                        # Find all links
                        links = soup.find_all('a', href=True)
                        same_domain_links = []
                        
                        # Filter links to same domain only
                        for link in links:
                            href = link['href']
                            abs_url = urljoin(url, href)
                            parsed_link = urlparse(abs_url)
                            
                            # Only include links from the same domain
                            if parsed_link.netloc == base_domain:
                                same_domain_links.append(abs_url)
                        
                        # Process up to 10 links from same domain
                        unique_links = list(set(same_domain_links))[:10]
                        logger.info(f"{depth_indicator}Found {len(links)} total links, {len(same_domain_links)} same-domain links, processing {len(unique_links)} unique links")
                        
                        # Recursively process each link
                        for i, link in enumerate(unique_links):
                            if link not in processed_urls:
                                logger.info(f"{depth_indicator}Processing link {i+1}/{len(unique_links)}: {link}")
                                processed_count += process_url_recursive(
                                    link, current_depth + 1, max_depth)
                    except Exception as e:
                        error_msg = str(e)
                        logger.error(f"{depth_indicator}Error extracting links from {current_url}: {error_msg}")
                        
                return processed_count
            except Exception as e:
                error_msg = str(e)
                logger.error(f"{depth_indicator}Error processing URL {current_url}: {error_msg}")
                return 0
        
        # Check if max_depth is specified in metadata
        max_depth = 0
        if metadata and "max_depth" in metadata:
            max_depth = int(metadata["max_depth"])
        
        logger.info(f"Starting recursive URL processing with max_depth={max_depth}")
        
        # Process the URL recursively
        total_processed = process_url_recursive(url, 0, max_depth)
        
        logger.info(f"URL processing complete. Processed {total_processed} total pages.")
        
        if total_processed > 1:
            return f"Successfully processed {url} and {total_processed-1} linked pages."
        else:
            return f"Successfully processed {url}."
            
    def add_recursive_url_content(self, base_url: str, max_depth: int = 2, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Add content from a URL and its linked pages recursively to the knowledge base.
        
        Args:
            base_url: Base URL to fetch and learn from
            max_depth: Maximum depth for recursive crawling
            metadata: Optional metadata about the content
            
        Returns:
            Status message
        """
        try:
            # Load the content from the URL recursively
            loader = RecursiveUrlLoader(
                url=base_url,
                max_depth=max_depth,
                extractor=lambda x: x.split("<body>")[1].split("</body>")[0]
            )
            documents = loader.load()
            
            # Add metadata if provided
            if metadata:
                for doc in documents:
                    doc.metadata.update(metadata)
                    
            # Add source and timestamp to metadata
            for doc in documents:
                doc.metadata["source"] = doc.metadata.get("source", base_url)
                doc.metadata["timestamp"] = str(documents[0].metadata.get("last_modified", ""))
                
            # Split the documents
            split_docs = self.text_splitter.split_documents(documents)
            
            # Add the documents to the knowledge base
            result = self.knowledge_base.add_documents(split_docs)
            
            return f"Successfully processed {base_url} and {len(documents)} linked pages: {result}"
        except Exception as e:
            return f"Error processing {base_url}: {str(e)}"
    
    def add_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Add text directly to the knowledge base.
        
        Args:
            text: Text to add
            metadata: Optional metadata about the text
            
        Returns:
            Status message
        """
        try:
            # Create a document from the text
            doc = Document(page_content=text, metadata=metadata or {})
            
            # Split the document
            split_docs = self.text_splitter.split_documents([doc])
            
            # Add the documents to the knowledge base
            result = self.knowledge_base.add_documents(split_docs)
            
            return f"Successfully added text: {result}"
        except Exception as e:
            return f"Error adding text: {str(e)}"
