"""
Document retriever implementation for the AI Research Agent.
This module handles retrieving documents from various sources and integrating with the knowledge base.
"""

import os
from typing import List, Dict, Any, Optional
import tempfile
import requests
from datetime import datetime
from bs4 import BeautifulSoup
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
                - use_js: Optional[bool] - whether to use JavaScript rendering for content extraction
            
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
                
                # Check if enhanced extraction is requested
                use_enhanced = metadata.get("use_js", False) if metadata else False
                
                # Variables to store the response and BeautifulSoup object for reuse
                response = None
                soup = None
                
                if use_enhanced:
                    try:
                        logger.info(f"{depth_indicator}Using enhanced content extraction for {current_url}")
                        # Custom extraction with improved BeautifulSoup parsing
                        response = requests.get(current_url, headers={
                            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                        }, timeout=30)
                        soup = BeautifulSoup(response.content, 'html.parser')
                        
                        # Keep a copy of the full soup for link extraction later
                        full_soup = BeautifulSoup(response.content, 'html.parser')  
                        
                        # Remove navigation elements, ads, footers, etc. for content extraction only
                        for element in soup.select('nav, header, footer, .ads, .navigation, .menu, #menu, .sidebar, .comment, .comments, script, style'):
                            element.decompose()
                        
                        # Extract main content more aggressively
                        main_content = soup.select_one('main, #main, .main, article, .article, .content, #content, .post, .post-content')
                        text = main_content.get_text(separator='\n', strip=True) if main_content else soup.get_text(separator='\n', strip=True)
                        
                        # Create document from extracted text
                        doc = Document(page_content=text, metadata={"source": current_url})
                        documents = [doc]
                        
                        # Use the full soup for link extraction later
                        soup = full_soup
                        
                        logger.info(f"{depth_indicator}Enhanced extraction complete, extracted {len(text)} characters")
                    except Exception as enhanced_error:
                        logger.warning(f"{depth_indicator}Enhanced extraction failed: {str(enhanced_error)}. Falling back to basic loader.")
                        loader = WebBaseLoader(current_url)
                        documents = loader.load()
                else:
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
                        
                        # Only make a new request if we don't already have a soup object
                        if soup is None:
                            logger.info(f"{depth_indicator}Making new request to extract links...")
                            response = requests.get(current_url, headers={
                                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                            })
                            soup = BeautifulSoup(response.content, 'html.parser')
                        else:
                            logger.info(f"{depth_indicator}Reusing existing soup object for link extraction")
                        
                        # Find all links
                        links = soup.find_all('a', href=True)
                        
                        logger.info(f"{depth_indicator}Found {len(links)} <a> tags with href attributes")
                        
                        # Process all links without domain restriction
                        all_links = []
                        filtered_out = 0
                        wiki_links = 0
                        for link in links:
                            href = link['href']
                            
                            # Convert relative URLs to absolute
                            abs_url = urljoin(current_url, href)
                            parsed_link = urlparse(abs_url)
                            
                            # Debug specific links
                            if 'wikipedia' in abs_url:
                                wiki_links += 1
                                if wiki_links <= 5:  # Only log a few for debug purposes
                                    logger.info(f"{depth_indicator}Found Wikipedia link: {abs_url}")
                            
                            # Skip unwanted links - improved filtering
                            if parsed_link.scheme not in ['http', 'https']:
                                filtered_out += 1
                                continue
                                
                            # Skip fragments of the same page and other common cases to avoid
                            if abs_url == current_url or \
                               '#' in abs_url or \
                               'Special:' in abs_url or \
                               'action=edit' in abs_url or \
                               'File:' in abs_url:
                                filtered_out += 1
                                continue
                                
                            # Keep this link
                            all_links.append(abs_url)
                            
                        # Remove duplicates but don't limit the number of links
                        unique_links = list(set(all_links))
                        
                        logger.info(f"{depth_indicator}Found {len(links)} total links, filtered out {filtered_out}, keeping {len(unique_links)} unique links for processing")
                        
                        # Only recursively process links if we haven't reached max depth
                        if current_depth < max_depth:
                            logger.info(f"{depth_indicator}Current depth {current_depth} is less than max_depth {max_depth}, will process links recursively")
                            
                            # Recursively process each link
                            for i, link in enumerate(unique_links):
                                if link not in processed_urls:
                                    logger.info(f"{depth_indicator}Processing link {i+1}/{len(unique_links)}: {link}")
                                    link_processed = process_url_recursive(
                                        link, current_depth + 1, max_depth)
                                    logger.info(f"{depth_indicator}Finished processing {link}, processed {link_processed} pages")
                                    processed_count += link_processed
                                else:
                                    logger.info(f"{depth_indicator}Skipping already processed link: {link}")
                        else:
                            logger.info(f"{depth_indicator}Reached max depth {max_depth}, not following links from this page")
                        
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
            logger.info(f"Using max_depth={max_depth} from metadata")
        else:
            logger.warning(f"No max_depth specified in metadata, recursion will not happen")
        
        logger.info(f"Starting recursive URL processing with max_depth={max_depth}")
        
        # Debug the processed_urls set
        logger.info(f"Initial processed_urls set is empty: {len(processed_urls) == 0}")
        
        # Process the URL recursively
        total_processed = process_url_recursive(url, 0, max_depth)
        
        logger.info(f"Final processed_urls set size: {len(processed_urls)}")
        logger.info(f"URL processing complete. Processed {total_processed} total pages.")
        
        if total_processed > 1:
            return f"Successfully processed {url} and {total_processed-1} linked pages."
        else:
            return f"Successfully processed {url}."
            
    def add_recursive_url_content(self, base_url: str, max_depth: int = 2, metadata: Optional[Dict[str, Any]] = None, excluded_domains: Optional[List[str]] = None) -> str:
        """
        Add content from a URL and its linked pages recursively to the knowledge base.
        
        Args:
            base_url: Base URL to fetch and learn from
            max_depth: Maximum depth for recursive crawling
            metadata: Optional metadata about the content
            excluded_domains: Optional list of domains to exclude from crawling (e.g., ['books.google.com'])
            
        Returns:
            Status message
        """
        import base64
        import os
        import re
        import json
        import hashlib
        from typing import Optional, Dict, Any, List, Union, Tuple
        from urllib.parse import urlparse, urljoin
        import time
        import logging
        from pathlib import Path
        from dotenv import load_dotenv

        # Load environment variables
        load_dotenv()
        
        logger = logging.getLogger(__name__)
        
        # Make sure we have metadata dict
        if metadata is None:
            metadata = {}
            
        use_enhanced = metadata.get("use_js", False)
        enhanced_str = "enhanced" if use_enhanced else "standard"
        
        logger.info(f"Starting {enhanced_str} recursive URL crawling for {base_url} with max_depth={max_depth}")
        
        # Global tracking of processed URLs
        global_processed_urls = set()
        all_documents = []
        user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/93.0.4577.63 Safari/537.36"
        
        # Get configurable crawling limits from environment variables first, then metadata, then defaults
        default_max_urls = int(os.getenv("MAX_CRAWL_URLS", 100))
        default_max_urls_per_page = int(os.getenv("MAX_CRAWL_URLS_PER_PAGE", 50))
        
        # Metadata values override environment variables if present
        max_urls_total = metadata.get("max_urls", default_max_urls)
        max_urls_per_page = metadata.get("max_urls_per_page", default_max_urls_per_page)
        
        logger.info(f"Using crawling limits: max_urls_total={max_urls_total}, max_urls_per_page={max_urls_per_page}")
        logger.info(f"(Configure these globally in .env with MAX_CRAWL_URLS and MAX_CRAWL_URLS_PER_PAGE)")
        
        
        def process_url(url, depth):
            """Process a URL and extract its content and links"""
            # Check if URL should be excluded based on domain
            if url in global_processed_urls:
                logger.info(f"{'  ' * depth}[{depth}] Skipping already processed URL: {url}")
                return None
                
            parsed_url = urlparse(url)
            domain = parsed_url.netloc
            
            # Skip excluded domains
            if excluded_domains and any(exc_domain in domain for exc_domain in excluded_domains):
                logger.info(f"{'  ' * depth}[{depth}] Skipping excluded domain: {domain}")
                global_processed_urls.add(url)  # Mark as processed so we don't try again
                return None
            
            try:
                global_processed_urls.add(url)
                logger.info(f"{'  ' * depth}[{depth}] Processing URL: {url}")
                
                # Make the request
                headers = {"User-Agent": user_agent}
                response = requests.get(url, headers=headers, timeout=10)
                response.raise_for_status()
                
                # Parse with BeautifulSoup
                soup = BeautifulSoup(response.content, "html.parser")
                
                # Extract the content
                page_content = ""
                if use_enhanced:
                    # More aggressively filter out navigation, ads, etc.
                    # Remove script and style elements
                    for element in soup(["script", "style", "nav", "footer", "header", "aside"]):
                        element.extract()
                    
                    # Extract text from main content areas
                    content_tags = soup.select("article, main, div.content, div#content, div.main, div#main")
                    if content_tags:
                        for tag in content_tags:
                            page_content += tag.get_text(separator=" ", strip=True) + "\n\n"
                    else:
                        # Fallback to body content if no main content identified
                        if soup.body:
                            page_content = soup.body.get_text(separator=" ", strip=True)
                        else:
                            page_content = soup.get_text(separator=" ", strip=True)
                else:
                    # Basic extraction of all text
                    page_content = soup.get_text(separator=" ", strip=True)
                
                # Create a document
                title = soup.title.string if soup.title else url
                doc = Document(
                    page_content=page_content,
                    metadata={
                        "source": url,
                        "title": title,
                        "document_id": url,  # Use URL as document ID
                        "content_hash": hashlib.sha256(page_content.encode()).hexdigest(),
                        "content_length": len(page_content),
                        "depth": depth,
                        "timestamp": datetime.now().isoformat()
                    }
                )
                
                # Add any user metadata
                if metadata:
                    for key, value in metadata.items():
                        if key != "max_depth":  # Skip max_depth as it's not document metadata
                            doc.metadata[key] = value
                
                # Add document to the list
                all_documents.append(doc)
                logger.info(f"{'  ' * depth}[{depth}] Added document for {url} with {len(page_content)} characters")
                
                # Extract links for recursion
                extracted_links = []
                if depth < max_depth:
                    links = soup.find_all('a', href=True)
                    for link in links:
                        href = link['href']
                        abs_url = urljoin(url, href)
                        parsed = urlparse(abs_url)
                        
                        # Filter out unwanted URLs
                        if parsed.scheme in ['http', 'https'] and \
                           abs_url != url and \
                           '#' not in abs_url and \
                           'javascript:' not in href and \
                           not any(skip in abs_url for skip in ['Special:', 'action=edit', 'File:']):
                            extracted_links.append(abs_url)
                    
                    # Remove duplicates
                    extracted_links = list(set(extracted_links))
                    logger.info(f"{'  ' * depth}[{depth}] Found {len(extracted_links)} unique links to follow")
                
                return extracted_links
                
            except Exception as e:
                logger.error(f"{'  ' * depth}[{depth}] Error processing {url}: {str(e)}")
                return None
        
        def crawl_recursive(url, depth=0):
            """Recursively crawl URLs up to max_depth"""
            # Stop if we've reached beyond max_depth
            if depth > max_depth:
                return
            
            # Process the current URL
            links = process_url(url, depth)
            
            # Only follow links if we're still below max_depth
            if links and depth < max_depth:
                logger.info(f"{'  ' * depth}[{depth}] Found {len(links)} links at depth {depth}, max_depth={max_depth}")
                
                # With max_depth=1, we only process the original page (depth=0)
                # and its immediate links (depth=1), but don't go deeper
                links_to_process = len(links)
                processed = 0
                
                for link in links:
                    # Apply configurable safety limits: max_urls_total across all pages and max_urls_per_page per page
                    if link not in global_processed_urls and len(global_processed_urls) < max_urls_total and processed < max_urls_per_page:
                        logger.info(f"{'  ' * depth}[{depth}] Processing link {processed+1} of {links_to_process} (limited to {max_urls_per_page}): {link}")
                        processed += 1
                        # Important: depth+1 here means links from the original page will be at depth=1
                        # With max_depth=1, this means we don't go any deeper
                        crawl_recursive(link, depth + 1)
                    elif processed >= max_urls_per_page:
                        logger.info(f"{'  ' * depth}[{depth}] Reached max_urls_per_page limit ({max_urls_per_page}) for this page")
                        break
                    elif len(global_processed_urls) >= max_urls_total:
                        logger.info(f"{'  ' * depth}[{depth}] Reached max_urls_total limit ({max_urls_total}) across all pages")
                        break
                    
        try:
            # Start the recursive crawling
            crawl_recursive(base_url)
            
            # Process all documents in batch
            logger.info(f"Crawling complete. Processed {len(global_processed_urls)} URLs with {len(all_documents)} documents")
            
            # Split the documents
            split_docs = self.text_splitter.split_documents(all_documents)
            logger.info(f"Split into {len(split_docs)} chunks for the knowledge base")
            
            # Add to knowledge base in batches to avoid ChromaDB limitations
            # Max batch size for ChromaDB is ~5000 items
            batch_size = 500  # Conservative batch size
            total_added = 0
            total_skipped = 0
            
            logger.info(f"Adding documents to knowledge base in batches of {batch_size}")
            for i in range(0, len(split_docs), batch_size):
                batch = split_docs[i:i+batch_size]
                logger.info(f"Processing batch {i//batch_size + 1}/{(len(split_docs)-1)//batch_size + 1} with {len(batch)} documents")
                try:
                    result = self.knowledge_base.add_documents(batch)
                    # Parse result string for counts
                    if isinstance(result, str) and "added" in result:
                        try:
                            # More robust parsing that handles different formats
                            if "added " in result:
                                added_parts = result.split("added ")[1].split(",")
                                if added_parts and added_parts[0]:
                                    added_str = added_parts[0].strip()
                                    added = int(added_str) if added_str.isdigit() else 0
                                    total_added += added
                            
                            if "skipped " in result:
                                skipped_parts = result.split("skipped ")[1].split(" duplicates")
                                if skipped_parts and skipped_parts[0]:
                                    skipped_str = skipped_parts[0].strip()
                                    skipped = int(skipped_str) if skipped_str.isdigit() else 0
                                    total_skipped += skipped
                        except Exception as e:
                            logger.error(f"Error parsing result string: {str(e)}")
                            # Continue processing anyway since this is just for reporting
                    logger.info(f"Batch result: {result}")
                except Exception as e:
                    logger.error(f"Error adding batch to knowledge base: {str(e)}")
            
            return f"Successfully processed {base_url} recursively. Crawled {len(global_processed_urls)} pages, created {len(all_documents)} documents, split into {len(split_docs)} chunks. Added {total_added} to knowledge base, skipped {total_skipped} duplicates."
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error in recursive URL processing: {error_msg}")
            return f"Error processing {base_url} recursively: {error_msg}"

    
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
