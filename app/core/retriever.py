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
            
        Returns:
            Status message
        """
        try:
            # Load the content from the URL
            loader = WebBaseLoader(url)
            documents = loader.load()
            
            # Add metadata if provided
            if metadata:
                for doc in documents:
                    doc.metadata.update(metadata)
                    
            # Add source and timestamp to metadata
            for doc in documents:
                doc.metadata["source"] = url
                doc.metadata["timestamp"] = str(documents[0].metadata.get("last_modified", ""))
                
            # Split the documents
            split_docs = self.text_splitter.split_documents(documents)
            
            # Add the documents to the knowledge base
            result = self.knowledge_base.add_documents(split_docs)
            
            return f"Successfully processed {url}: {result}"
        except Exception as e:
            return f"Error processing {url}: {str(e)}"
            
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
