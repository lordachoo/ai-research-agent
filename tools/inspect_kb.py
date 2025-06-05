# Knowledge Base Inspector Script
# Shows all chunks for a specific document_id or lists all document_ids

import sys
import os
from typing import List, Dict, Any, Optional
import json
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
from app.core.knowledge_base import KnowledgeBase

def pretty_print(data):
    """Print data in a readable format"""
    if isinstance(data, list):
        for i, item in enumerate(data):
            print(f"\n--- Item {i+1} ---")
            if isinstance(item, dict):
                for k, v in item.items():
                    if k == "page_content":
                        print(f"{k}:\n{v}")
                    else:
                        print(f"{k}: {v}")
            else:
                print(item)
    elif isinstance(data, dict):
        for k, v in data.items():
            print(f"{k}: {v}")

def list_document_ids(kb: KnowledgeBase) -> List[str]:
    """List all document IDs in the knowledge base"""
    collection_data = kb.vector_store.get()
    unique_docs = set()
    
    if "metadatas" in collection_data and collection_data["metadatas"]:
        for metadata in collection_data["metadatas"]:
            if metadata and "document_id" in metadata:
                unique_docs.add(metadata["document_id"])
    
    return sorted(list(unique_docs))

def get_chunks_for_document(kb: KnowledgeBase, document_id: str) -> List[Dict[str, Any]]:
    """Get all chunks for a specific document ID"""
    collection_data = kb.vector_store.get()
    chunks = []
    
    if "metadatas" in collection_data and collection_data["metadatas"]:
        for i, metadata in enumerate(collection_data["metadatas"]):
            if metadata and "document_id" in metadata and metadata["document_id"] == document_id:
                chunks.append({
                    "page_content": collection_data["documents"][i] if i < len(collection_data["documents"]) else "No content",
                    "metadata": metadata
                })
    
    return chunks

def main():
    # Load environment variables
    load_dotenv()
    
    # Initialize knowledge base
    kb_dir = os.environ.get("KNOWLEDGE_BASE_DIR", "./knowledge_base")
    kb = KnowledgeBase(persist_directory=kb_dir)
    
    # Command logic
    if len(sys.argv) == 1:
        # List all document IDs
        document_ids = list_document_ids(kb)
        print(f"Found {len(document_ids)} documents in knowledge base:")
        for doc_id in document_ids:
            print(f"- {doc_id}")
        print("\nTo view chunks for a specific document, run:")
        print("python inspect_kb.py <document_id>")
        
    elif len(sys.argv) >= 2:
        # Get document ID from command line argument
        document_id = sys.argv[1]
        
        # Get all chunks for the document
        chunks = get_chunks_for_document(kb, document_id)
        
        print(f"Found {len(chunks)} chunks for document '{document_id}':")
        
        # Check if we should output to file
        if len(sys.argv) > 2 and sys.argv[2] == "--save":
            output_file = f"{document_id.replace('/', '_').replace(':', '_')}_chunks.json"
            with open(output_file, "w") as f:
                json.dump(
                    [{"content": c["page_content"], "metadata": c["metadata"]} for c in chunks],
                    f, indent=2
                )
            print(f"Saved chunks to {output_file}")
        else:
            # Print details of each chunk
            for i, chunk in enumerate(chunks):
                print(f"\n--- Chunk {i+1}/{len(chunks)} ---")
                print(f"Content ({len(chunk['page_content'])} chars):")
                print(chunk["page_content"])
                print("\nMetadata:")
                pretty_print(chunk["metadata"])

if __name__ == "__main__":
    main()
