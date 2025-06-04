"""
Main entry point for the AI Research Agent.
This module provides command-line functionality for interacting with the agent.
"""

import os
# Forcefully disable LangChain tracing to suppress API key warning
os.environ['LANGCHAIN_TRACING_V2'] = 'false'
import sys
import argparse
import os
import json
import logging
from typing import Dict, Any, Optional, List
import dotenv
dotenv.load_dotenv() # Load environment variables from .env file

from app.core.agent import ResearchAgent
from app.core.knowledge_base import KnowledgeBase
from app.schedulers.source_scheduler import SourceScheduler

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


def setup_agent() -> ResearchAgent:
    """
    Set up the agent with the knowledge base.
    
    Returns:
        ResearchAgent instance
    """
    # Environment variables are loaded at the top of the script
    # Get configuration from environment variables
    kb_dir = os.getenv("KNOWLEDGE_BASE_DIR", "./knowledge_base")
    model_name = os.getenv("MODEL_NAME", "gpt-3.5-turbo-0125")
    agent_name = os.getenv("AGENT_NAME", "Research Assistant")
    temperature = float(os.getenv("TEMPERATURE", "0.1"))
    llm_provider = os.getenv("LLM_PROVIDER", "openai").lower()
    ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    # For OpenAI, MODEL_NAME is used directly. For Ollama, ResearchAgent will use OLLAMA_MODEL_NAME.
    openai_model_name = os.getenv("MODEL_NAME", "gpt-3.5-turbo-0125") 
    ollama_chat_model_name = os.getenv("OLLAMA_MODEL_NAME", "llama3:latest")

    # Initialize knowledge base (it reads its own env vars for embeddings)
    logger.info(f"Initializing knowledge base at {kb_dir} using {llm_provider} for embeddings (if applicable)")
    kb = KnowledgeBase(persist_directory=kb_dir, embedding_model=openai_model_name) # openai_model_name is default if not ollama
    
    # Determine model name for logging based on provider
    display_model_name = ollama_chat_model_name if llm_provider == "ollama" else openai_model_name

    # Initialize agent
    logger.info(f"Creating agent '{agent_name}' with {llm_provider} provider and model {display_model_name}")
    agent = ResearchAgent(
        agent_name=agent_name,
        knowledge_base=kb,
        model_name=openai_model_name,  # Pass OpenAI model name, agent will pick ollama model if provider is ollama
        temperature=temperature,
        llm_provider=llm_provider,
        ollama_base_url=ollama_base_url,
        ollama_model_name=ollama_chat_model_name
    )
    
    return agent


def query_agent(agent: ResearchAgent, query: str, chat_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
    """
    Query the agent with a question.
    
    Args:
        agent: ResearchAgent instance
        query: Query string
        chat_history: Optional chat history
        
    Returns:
        Agent response
    """
    logger.info(f"Querying agent: {query}")
    response = agent.run(query, chat_history)
    return response


def add_document(agent: ResearchAgent, document_path: str, metadata: Optional[Dict[str, Any]] = None) -> str:
    """
    Add a document to the agent's knowledge base.
    
    Args:
        agent: ResearchAgent instance
        document_path: Path to the document
        metadata: Optional metadata
        
    Returns:
        Status message
    """
    logger.info(f"Adding document: {document_path}")
    result = agent.learn_from_document(document_path, metadata)
    return result


def add_url(agent: ResearchAgent, url: str, max_depth: int = 0, metadata: Optional[Dict[str, Any]] = None) -> str:
    """
    Add content from a URL to the agent's knowledge base.
    
    Args:
        agent: ResearchAgent instance
        url: URL to fetch and learn from
        max_depth: Depth for recursive URL fetching
        metadata: Optional metadata
        
    Returns:
        Status message
    """
    logger.info(f"Adding URL: {url}")
    if max_depth > 0:
        result = agent.retriever.add_recursive_url_content(url, max_depth=max_depth, metadata=metadata)
    else:
        result = agent.learn_from_url(url, metadata=metadata)
    return result


def schedule_source(agent: ResearchAgent, config_path: str) -> Dict[str, str]:
    """
    Schedule sources from a configuration file.
    
    Args:
        agent: ResearchAgent instance
        config_path: Path to the configuration file
        
    Returns:
        Dictionary of source IDs and status messages
    """
    logger.info(f"Scheduling sources from: {config_path}")
    
    # Read the configuration file
    with open(config_path, 'r') as f:
        config = json.load(f)
        
    # Initialize the scheduler
    scheduler = SourceScheduler(research_agent=agent)
    
    results = {}
    
    # Add each source for knowledge base population
    for source_config in config.get("sources", []):
        source_id = source_config.get("source_id")
        source_type = source_config.get("source_type")
        
        if not source_id or not source_type:
            logger.error(f"Invalid source configuration: {source_config}")
            results[f"invalid_source_{len(results)}"] = "Invalid source configuration"
            continue
            
        if source_type == "url":
            url = source_config.get("url")
            if not url:
                logger.error(f"Missing URL for source: {source_id}")
                results[source_id] = "Missing URL for source"
                continue
            interval_minutes = source_config.get("interval_minutes", 60)
            metadata = source_config.get("metadata")
            max_depth = source_config.get("max_depth", 0)
            result = scheduler.add_url_source(
                source_id=source_id, url=url, interval_minutes=interval_minutes, metadata=metadata, max_depth=max_depth
            )
        elif source_type == "arxiv":
            search_query = source_config.get("search_query")
            if not search_query:
                logger.error(f"Missing search query for arXiv source: {source_id}")
                results[source_id] = "Missing search query for arXiv source"
                continue
            categories = source_config.get("categories")
            interval_hours = source_config.get("interval_hours", 24)
            max_results = source_config.get("max_results", 10)
            metadata = source_config.get("metadata")
            result = scheduler.add_arxiv_source(
                source_id=source_id, search_query=search_query, categories=categories, 
                interval_hours=interval_hours, max_results=max_results, metadata=metadata
            )
        else:
            logger.error(f"Unsupported source type for knowledge base: {source_type}")
            results[source_id] = f"Unsupported source type: {source_type}"
            continue
        results[source_id] = result

    # Add each scheduled task (e.g., prompt on URL)
    for task_config in config.get("scheduled_tasks", []):
        task_id = task_config.get("task_id")
        task_type = task_config.get("task_type")

        if not task_id or not task_type:
            logger.error(f"Invalid task configuration: {task_config}")
            results[f"invalid_task_{len(results)}"] = "Invalid task configuration"
            continue

        if task_type == "prompt_on_url":
            url = task_config.get("url")
            prompt_template = task_config.get("prompt_template")
            output_action = task_config.get("output_action")

            if not url or not prompt_template or not output_action:
                logger.error(f"Missing required fields (url, prompt_template, or output_action) for task: {task_id}")
                results[task_id] = "Missing required fields for prompt_on_url task"
                continue
            
            interval_minutes = task_config.get("interval_minutes", 60)
            cron_expression = task_config.get("cron_expression") # Can be None
            metadata = task_config.get("metadata") # Can be None

            result = scheduler.add_prompt_on_url_task(
                task_id=task_id,
                url=url,
                prompt_template=prompt_template,
                output_action=output_action,
                interval_minutes=interval_minutes,
                cron_expression=cron_expression,
                metadata=metadata
            )
        else:
            logger.error(f"Unsupported task type: {task_type}")
            results[task_id] = f"Unsupported task type: {task_type}"
            continue
        results[task_id] = result
        
    # Keep the scheduler running
    logger.info("Sources scheduled. Press Ctrl+C to stop.")
    
    try:
        # Print source and task information
        logger.info("--- Scheduled Sources for Knowledge Base ---")
        for source_id, source_info in scheduler.get_all_sources().items():
            logger.info(f"  Source ID: {source_id}, Type: {source_info.get('type')}, Details: {source_info.get('url') or source_info.get('search_query')}, Schedule: {source_info.get('schedule_info')}")
        
        logger.info("--- Scheduled Tasks (Prompts) ---")
        for task_id, task_info in scheduler.get_all_scheduled_tasks().items():
            logger.info(f"  Task ID: {task_id}, Type: {task_info.get('type')}, URL: {task_info.get('url')}, Schedule: {task_info.get('schedule_info')}")
            
        # Keep the main thread running
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down scheduler...")
        scheduler.shutdown()
        
    return results


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="AI Research Agent")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Query command
    query_parser = subparsers.add_parser("query", help="Query the agent")
    query_parser.add_argument("query", help="Query string")
    
    # Add document command
    doc_parser = subparsers.add_parser("add-document", help="Add a document to the knowledge base")
    doc_parser.add_argument("document_path", help="Path to the document")
    doc_parser.add_argument("--metadata", help="JSON metadata about the document")
    
    # Add URL command
    url_parser = subparsers.add_parser("add-url", help="Add content from a URL to the knowledge base")
    url_parser.add_argument("url", help="URL to fetch and learn from")
    url_parser.add_argument("--depth", type=int, default=0, help="Depth for recursive URL fetching")
    url_parser.add_argument("--metadata", help="JSON metadata about the content")
    
    # Schedule sources command
    schedule_parser = subparsers.add_parser("schedule", help="Schedule sources from a configuration file")
    schedule_parser.add_argument("config_path", help="Path to the configuration file")
    
    # Start API command
    api_parser = subparsers.add_parser("api", help="Start the API server")
    api_parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    api_parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    if not args.command:
        print("No command specified. Use --help for usage information.")
        return
    
    # Set up the agent
    agent = setup_agent()
    
    # Execute the command
    if args.command == "query":
        response = query_agent(agent, args.query)
        print("\nAgent Response:")
        print(response["output"])
        
    elif args.command == "add-document":
        doc_path = args.document_path
        logger.info(f"Processing path for document addition: {doc_path}")

        if doc_path.startswith("http://") or doc_path.startswith("https://"):
            logger.info(f"Adding URL: {doc_path}")
            result = agent.learn_from_url(doc_path)
            print(result)
        elif os.path.isdir(doc_path):
            logger.info(f"Adding documents from directory: {doc_path}")
            added_files = 0
            skipped_files = 0
            for filename in os.listdir(doc_path):
                file_path = os.path.join(doc_path, filename)
                if os.path.isfile(file_path):
                    try:
                        logger.info(f"Adding document file: {file_path}")
                        result = agent.learn_from_document(file_path)
                        print(f"Successfully added {file_path}: {result}")
                        added_files += 1
                    except Exception as e:
                        print(f"Failed to add {file_path}: {e}")
                        skipped_files += 1
                else:
                    logger.info(f"Skipping non-file item: {file_path}")
            print(f"\nDirectory processing complete. Added {added_files} files, skipped {skipped_files} files/items.")
        elif os.path.isfile(doc_path):
            logger.info(f"Adding document file: {doc_path}")
            result = agent.learn_from_document(doc_path)
            print(result)
        else:
            print(f"Error: Document path '{doc_path}' is not a valid file, directory, or URL.")
            
    elif args.command == "add-url":
        metadata = json.loads(args.metadata) if args.metadata else None
        result = add_url(agent, args.url, args.depth, metadata)
        print(result)
        
    elif args.command == "schedule":
        results = schedule_source(agent, args.config_path)
        for source_id, result in results.items():
            print(f"{source_id}: {result}")
            
    elif args.command == "api":
        import uvicorn
        from app.api import app
        
        print(f"Starting API server on {args.host}:{args.port}")
        uvicorn.run("app.api:app", host=args.host, port=args.port, reload=True)
    

if __name__ == "__main__":
    main()
