"""
Web UI launcher for the AI Research Agent.
This module provides a function to start the FastAPI web interface.
"""

import os
import sys
import logging
import uvicorn
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def launch_web_ui(host="0.0.0.0", port=8000, reload=False, log_level="info"):
    """
    Launch the FastAPI web UI for the AI Research Agent.
    
    Args:
        host (str): Host address to bind the server to
        port (int): Port to run the server on
        reload (bool): Whether to reload the server on code changes
        log_level (str): Log level for uvicorn
    """
    # Load environment variables
    load_dotenv()
    
    # Log configuration
    logger.info(f"Starting FastAPI web UI on http://{host}:{port}")
    logger.info(f"Using knowledge base directory: {os.getenv('KNOWLEDGE_BASE_DIR', './knowledge_base')}")
    
    # Determine the correct model name to display based on provider
    llm_provider = os.getenv("LLM_PROVIDER", "").lower()
    if llm_provider == "ollama":
        model_name = os.getenv("OLLAMA_MODEL_NAME", "Not specified")
        logger.info(f"Using LLM provider: {llm_provider}")
        logger.info(f"Using Ollama model: {model_name}")
    else:
        model_name = os.getenv("MODEL_NAME", "Not specified")
        logger.info(f"Using LLM provider: {llm_provider}")
        logger.info(f"Using model: {model_name}")
    
    # Start the server
    uvicorn.run(
        "app.api:app", 
        host=host,
        port=port,
        reload=reload,
        log_level=log_level
    )

if __name__ == "__main__":
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Launch the FastAPI web UI")
    parser.add_argument("--host", default="0.0.0.0", help="Host address to bind to")
    parser.add_argument("--port", default=8000, type=int, help="Port to run the server on")
    parser.add_argument("--reload", action="store_true", help="Reload server on code changes")
    parser.add_argument("--log-level", default="info", help="Log level for uvicorn")
    
    args = parser.parse_args()
    
    # Launch the web UI
    launch_web_ui(
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level
    )
