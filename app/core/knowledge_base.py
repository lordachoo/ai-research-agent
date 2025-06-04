"""
Knowledge base implementation for the AI Research Agent.
This module handles document storage, embeddings, and retrievals.
"""

import os
from typing import List, Dict, Any, Optional
import datetime
#from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document


class KnowledgeBase:
    """Knowledge base for storing and retrieving documents."""

    def __init__(
        self,
        persist_directory: str,
        embedding_model: str = "text-embedding-ada-002",
        collection_name: str = "research_documents"
    ):
        """
        Initialize the Knowledge Base.

        Args:
            persist_directory: Directory to persist the vector store
            embedding_model: Embedding model to use
            collection_name: Name of the collection in the vector store
        """
        self.persist_directory = persist_directory
        self.embedding_model_name = embedding_model # Retain for stats, actual model set below
        self.collection_name = collection_name
        self.llm_provider = os.getenv("LLM_PROVIDER", "openai").lower()
        self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.ollama_embedding_model_name = os.getenv("OLLAMA_EMBEDDING_MODEL_NAME", "nomic-embed-text:latest")

        # Create directory if it doesn't exist
        os.makedirs(persist_directory, exist_ok=True)

        # Initialize the embedding function based on provider
        if self.llm_provider == "ollama":
            self.embeddings = OllamaEmbeddings(
                model=self.ollama_embedding_model_name,
                base_url=self.ollama_base_url
            )
            self.embedding_model_name = self.ollama_embedding_model_name # Update for stats
        else: # Default to openai
            self.embeddings = OpenAIEmbeddings(model=embedding_model)
        
        # Initialize the vector store
        self.vector_store = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embeddings,
            collection_name=collection_name
        )
        
    def add_documents(self, documents: List[Document]) -> str:
        """
        Add documents to the knowledge base.
        
        Args:
            documents: List of documents to add
            
        Returns:
            Status message
        """
        import logging # Add logging import if not already present at top of file
        logger = logging.getLogger(__name__) # Or use self.logger if available

        docs_to_add_or_update = []
        added_count = 0
        updated_count = 0
        skipped_count = 0

        # Prepare a list of IDs from the input documents to efficiently check existence
        # and ensure each document has the necessary metadata.
        doc_ids_from_input = []
        valid_input_docs = []
        for doc in documents:
            doc_id = doc.metadata.get("document_id")
            if not doc_id:
                logger.warning(f"Document missing 'document_id' in metadata, cannot perform deduplication check. Adding it directly. Content preview: {doc.page_content[:100]}...")
                docs_to_add_or_update.append(doc)
                continue
            if "content_hash" not in doc.metadata:
                logger.warning(f"Document '{doc_id}' missing 'content_hash' in metadata. Adding it directly without hash check.")
                docs_to_add_or_update.append(doc)
                continue
            doc_ids_from_input.append(doc_id)
            valid_input_docs.append(doc)

        # Process each chunk
        for doc_chunk in valid_input_docs:
            original_url = doc_chunk.metadata["document_id"] # This is the URL
            new_summary_hash = doc_chunk.metadata["content_hash"] # Hash of the full summary

            # Check if any existing chunk in Chroma has the same original_url and new_summary_hash
            # This indicates the summary for this URL, with this exact content, has already been processed and stored.
            try:
                # Query for existing chunks that match both the URL and the full summary hash
                existing_chunks = self.vector_store.get(
                    where={
                        "$and": [
                            {"document_id": {"$eq": original_url}},
                            {"content_hash": {"$eq": new_summary_hash}}
                        ]
                    },
                    include=["metadatas"] # Not strictly needed for just existence check, but good for debug
                )

                if existing_chunks and existing_chunks['ids'] and len(existing_chunks['ids']) > 0:
                    logger.info(f"Summary for URL '{original_url}' with hash '{new_summary_hash}' already exists in DB. Skipping this chunk.")
                    skipped_count += 1
                    continue # Skip this chunk, as its summary is already present
                else:
                    # No existing chunk matches this specific URL *and* summary hash combination.
                    # This could be a new URL, or an existing URL with updated summary content.
                    # We need to check if there are *any* chunks for this URL, to see if it's an update or truly new.
                    older_version_chunks = self.vector_store.get(where={"document_id": {"$eq": original_url}})
                    if older_version_chunks and older_version_chunks['ids'] and len(older_version_chunks['ids']) > 0:
                        # This URL exists, but the summary hash is different. This is an UPDATE.
                        # We should ideally delete the old chunks for this URL before adding new ones.
                        logger.info(f"URL '{original_url}' exists with old summary. New summary hash '{new_summary_hash}'. Deleting old chunks and adding new.")
                        old_ids_to_delete = older_version_chunks['ids']
                        if old_ids_to_delete:
                            self.vector_store.delete(ids=old_ids_to_delete)
                            logger.info(f"Deleted {len(old_ids_to_delete)} old chunks for URL '{original_url}'.")
                        # This chunk contributes to an updated document, but we count unique URLs for 'updated_count' later if possible
                        # For now, let's increment updated_count if this is the first chunk triggering an update for this URL
                        # This simple increment might overcount if multiple chunks from the same updated doc are processed here.
                        # A more robust way is to track processed URLs for update status in this batch.
                        # For now, this will at least indicate an update event occurred.
                        if not any(d is doc_chunk for d in docs_to_add_or_update): # A simple check to see if we've decided to add this chunk already
                             updated_count += 1 # Increment per chunk that is part of an update for now
                    else:
                        # This URL is entirely new.
                        logger.info(f"URL '{original_url}' with summary hash '{new_summary_hash}' is new. Will add this chunk.")
                        # Similar to updated_count, this might overcount if not careful.
                        if not any(d is doc_chunk for d in docs_to_add_or_update):
                            added_count += 1 # Increment per chunk that is part of a new doc for now
                    
                    docs_to_add_or_update.append(doc_chunk)

            except Exception as e:
                logger.error(f"Error during Chroma get/delete for URL '{original_url}': {e}", exc_info=True)
                # Decide how to handle: skip this chunk or try to add it anyway?
                # For now, let's be cautious and skip if there's a DB error during check.
                skipped_count += 1
                continue

        try:
            if docs_to_add_or_update:
                self.vector_store.add_documents(docs_to_add_or_update)
                # The 'added_count' here reflects truly new additions based on ID.
                # 'updated_count' reflects items whose IDs existed but content changed.
                # Chroma's add_documents handles both cases if IDs are provided.
                # The counts added_count and updated_count here are more indicative of chunks processed for new/updated URLs.
                # A more accurate count of unique URLs added/updated would require tracking unique URLs processed in this batch.
                return f"Successfully processed document chunks. Chunks for new URLs: {added_count}, Chunks for updated URLs (old deleted, new added): {updated_count}, Chunks skipped (duplicate summary already exists): {skipped_count}."
            else:
                return f"No new or updated documents to add. Skipped {skipped_count} duplicates."
        except Exception as e:
            logger.error(f"Error adding documents to the vector store: {str(e)}", exc_info=True)
            return f"Error adding documents to the knowledge base: {str(e)}"
    
    def get_retriever(self, search_kwargs: Optional[Dict[str, Any]] = None):
        """
        Get a retriever for the knowledge base.
        
        Args:
            search_kwargs: Optional search parameters
            
        Returns:
            Retriever instance
        """
        if search_kwargs is None:
            search_kwargs = {"k": 5}
            
        return self.vector_store.as_retriever(search_kwargs=search_kwargs)
    
    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """
        Perform a similarity search in the knowledge base.
        
        Args:
            query: Query string
            k: Number of results to return
            
        Returns:
            List of relevant documents
        """
        return self.vector_store.similarity_search(query, k=k)
    
    def delete_collection(self) -> str:
        """
        Delete the collection in the vector store.
        
        Returns:
            Status message
        """
        try:
            self.vector_store.delete_collection()
            self.vector_store = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings,
                collection_name=self.collection_name
            )
            return "Successfully deleted the collection"
        except Exception as e:
            return f"Error deleting the collection: {str(e)}"
            
    def get_document_count(self) -> int:
        """
        Get the number of documents in the knowledge base.
        
        Returns:
            Number of documents
        """
        return len(self.vector_store.get()["ids"])
        
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the collection.
        
        Returns:
            Dictionary with collection statistics
        """
        try:
            collection_data = self.vector_store.get()
            stats = {
                "document_count": len(collection_data["ids"]),
                "collection_name": self.collection_name,
                "persist_directory": self.persist_directory,
                "embedding_model": self.embedding_model_name,
                "last_updated": datetime.datetime.now().isoformat()
            }
            return stats
        except Exception as e:
            return {"error": str(e)}
