#!/usr/bin/env python3
"""
LLM Ablation Test Script for Knowledge Base Toggle

This script compares responses from the ResearchAgent with and without 
knowledge base usage enabled, allowing for side-by-side comparison.
"""

import os
import sys
import argparse
import logging
from typing import List, Dict, Any, Optional
import time
from dotenv import load_dotenv

# Fix import paths to work from tools directory
# Add the parent directory (project root) to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Load environment variables from project root .env file
load_dotenv(os.path.join(project_root, ".env"))

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Now we can import from the app module
try:
    from app.core.agent import ResearchAgent
    from app.core.knowledge_base import KnowledgeBase
    logger.info("Successfully imported required modules")
except ImportError as e:
    logger.error(f"Failed to import required modules: {e}")
    logger.error("Make sure you're running this script from the ai-research-agent/tools directory")
    sys.exit(1)

def run_test(queries: List[str], show_diff: bool = True) -> None:
    """
    Run test queries with and without the knowledge base and compare results.
    
    Args:
        queries: List of queries to test
        show_diff: Whether to highlight differences between responses
    """
    # Initialize the knowledge base component
    logger.info("Initializing Knowledge Base...")
    kb_directory = os.path.join(project_root, "knowledge_base")
    logger.info(f"Using knowledge base directory: {kb_directory}")
    
    try:
        kb = KnowledgeBase(persist_directory=kb_directory)
        logger.info("Knowledge base initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize knowledge base: {e}")
        logger.error("Continuing with knowledge base disabled...")
        kb = None
    
    # Initialize the research agent
    logger.info("Initializing Research Agent...")
    try:
        agent = ResearchAgent(
            agent_name="KB Toggle Test Agent",
            knowledge_base=kb,
            # Use environment variables for LLM config or fallback to defaults
            llm_provider=os.environ.get("LLM_PROVIDER", "ollama"),
            ollama_model_name=os.environ.get("OLLAMA_MODEL_NAME", "llama3:latest"),
            temperature=float(os.environ.get("TEMPERATURE", "0.1"))
        )
        logger.info("Research agent initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize research agent: {e}")
        sys.exit(1)
        
    results = []
    
    # Process each query
    for i, query in enumerate(queries, 1):
        logger.info(f"Processing query {i}/{len(queries)}: '{query}'")
        
        # Store responses
        kb_response = None
        no_kb_response = None
        kb_time = 0
        no_kb_time = 0
        
        # Run with KB enabled
        logger.info("Running with Knowledge Base enabled...")
        try:
            start_time = time.time()
            kb_response = agent.run(query, use_knowledge_base=True)
            kb_time = time.time() - start_time
            logger.info(f"KB-enabled query completed in {kb_time:.2f} seconds")
        except Exception as e:
            logger.error(f"Error running KB-enabled query: {str(e)}")
            kb_response = f"ERROR: {str(e)}"
        
        # Run without KB
        logger.info("Running with Knowledge Base disabled...")
        try:
            start_time = time.time()
            no_kb_response = agent.run(query, use_knowledge_base=False)
            no_kb_time = time.time() - start_time
            logger.info(f"KB-disabled query completed in {no_kb_time:.2f} seconds")
        except Exception as e:
            logger.error(f"Error running KB-disabled query: {str(e)}")
            no_kb_response = f"ERROR: {str(e)}"
        
        # Store the results
        results.append({
            "query": query,
            "with_kb": {
                "response": kb_response,
                "time": kb_time
            },
            "without_kb": {
                "response": no_kb_response,
                "time": no_kb_time
            }
        })
    
    # Display results
    print("\n" + "="*80)
    print("Test Results:")
    print("="*80)
    
    for i, result in enumerate(results, 1):
        print(f"\nQuery {i}: {result['query']}")
        print("-"*80)
        
        print("\nResponse with Knowledge Base:")
        print(f"Time: {result['with_kb']['time']:.2f} seconds")
        print("-"*40)
        print(result['with_kb']['response'])
        
        print("\nResponse without Knowledge Base:")
        print(f"Time: {result['without_kb']['time']:.2f} seconds")
        print("-"*40)
        print(result['without_kb']['response'])
        
        # Calculate and show difference if requested
        if show_diff:
            kb_words = set(result['with_kb']['response'].lower().split())
            no_kb_words = set(result['without_kb']['response'].lower().split())
            
            unique_to_kb = kb_words - no_kb_words
            unique_to_no_kb = no_kb_words - kb_words
            
            if unique_to_kb or unique_to_no_kb:
                print("\nResponse Differences:")
                print("-"*40)
                if unique_to_kb:
                    print(f"Words unique to KB response: {', '.join(list(unique_to_kb)[:20])}")
                    if len(unique_to_kb) > 20:
                        print(f"... and {len(unique_to_kb) - 20} more")
                
                if unique_to_no_kb:
                    print(f"Words unique to non-KB response: {', '.join(list(unique_to_no_kb)[:20])}")
                    if len(unique_to_no_kb) > 20:
                        print(f"... and {len(unique_to_no_kb) - 20} more")
            else:
                print("\nThe responses are identical in terms of unique words.")
        
        print("\n" + "="*80)

def main():
    parser = argparse.ArgumentParser(description="Test LLM responses with and without Knowledge Base")
    
    parser.add_argument("-q", "--queries", nargs="+", 
                        help="List of queries to test", 
                        default=["What are gold mining techniques?",
                                "Explain quantum computing",
                                "What are the best practices for web development?"])
    
    parser.add_argument("--no-diff", action="store_true",
                        help="Don't show differences between responses")
    
    args = parser.parse_args()
    
    print("\nLLM Ablation Test - Knowledge Base Toggle\n")
    print(f"Testing {len(args.queries)} queries with and without Knowledge Base\n")
    
    run_test(args.queries, not args.no_diff)
    
    print("\nTest completed!")

if __name__ == "__main__":
    main()
