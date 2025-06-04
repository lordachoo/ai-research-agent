"""
Streamlit web interface for the AI Research Agent.
This module provides a web UI for interacting with the agent's core functionality.
"""

import os
import json
import streamlit as st
import sys
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Import the agent modules
from app.core.agent import ResearchAgent
from app.core.knowledge_base import KnowledgeBase
from app.schedulers.source_scheduler import SourceScheduler

# Setup page config
st.set_page_config(
    page_title="AI Research Agent",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "agent" not in st.session_state:
    st.session_state.agent = None
if "scheduler" not in st.session_state:
    st.session_state.scheduler = None
if "kb_stats" not in st.session_state:
    st.session_state.kb_stats = {"document_count": 0, "chunk_count": 0}
if "last_query" not in st.session_state:
    st.session_state.last_query = ""
if "last_response" not in st.session_state:
    st.session_state.last_response = ""


def setup_agent() -> ResearchAgent:
    """
    Set up the agent with the knowledge base.
    
    Returns:
        ResearchAgent instance
    """
    # Get configuration from environment variables
    kb_dir = os.getenv("KNOWLEDGE_BASE_DIR", "./knowledge_base")
    model_name = os.getenv("MODEL_NAME", "gpt-3.5-turbo-0125")
    agent_name = os.getenv("AGENT_NAME", "Research Assistant")
    temperature = float(os.getenv("TEMPERATURE", "0.1"))
    llm_provider = os.getenv("LLM_PROVIDER", "openai").lower()
    
    # Initialize knowledge base
    kb = KnowledgeBase(
        persist_directory=kb_dir
        # Note: llm_provider is taken from environment variables in KnowledgeBase class
    )
    
    # Set provider-specific variables
    if llm_provider == "ollama":
        ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        ollama_chat_model_name = os.getenv("OLLAMA_MODEL_NAME", "qwen2.5:14b")
        display_model_name = ollama_chat_model_name
    else:  # OpenAI
        ollama_base_url = None
        ollama_chat_model_name = None
        display_model_name = model_name
    
    # Initialize agent
    logger.info(f"Creating agent '{agent_name}' with {llm_provider} provider and model {display_model_name}")
    agent = ResearchAgent(
        agent_name=agent_name,
        knowledge_base=kb,
        model_name=model_name,
        temperature=temperature,
        llm_provider=llm_provider,
        ollama_base_url=ollama_base_url,
        ollama_model_name=ollama_chat_model_name
    )
    
    return agent


def get_kb_stats(agent: ResearchAgent) -> Dict[str, int]:
    """
    Get basic stats about the knowledge base.
    
    Args:
        agent: ResearchAgent instance
        
    Returns:
        Dictionary with stats
    """
    try:
        kb = agent.retriever.knowledge_base
        # These are approximations as ChromaDB doesn't provide direct count methods
        collection = kb.vector_store._collection
        doc_count = collection.count()
        # Distinct document_ids would require querying with distinct filter
        return {"document_count": doc_count, "chunk_count": doc_count}
    except Exception as e:
        logger.error(f"Error getting KB stats: {str(e)}")
        return {"document_count": 0, "chunk_count": 0}


def handle_query(agent: ResearchAgent, query: str) -> Dict[str, Any]:
    """
    Query the agent with a question.
    
    Args:
        agent: ResearchAgent instance
        query: Query string
        
    Returns:
        Agent response
    """
    try:
        st.session_state.last_query = query
        response = agent.run(query, st.session_state.chat_history)
        st.session_state.last_response = response["output"]
        
        # Update chat history
        st.session_state.chat_history.append({"role": "user", "content": query})
        st.session_state.chat_history.append({"role": "assistant", "content": response["output"]})
        
        return response
    except Exception as e:
        logger.error(f"Error querying agent: {str(e)}")
        return {"output": f"Error: {str(e)}", "error": True}


def handle_document_upload(agent: ResearchAgent, uploaded_file, metadata: Dict[str, Any] = None) -> str:
    """
    Process an uploaded document and add it to the knowledge base.
    
    Args:
        agent: ResearchAgent instance
        uploaded_file: The uploaded file from Streamlit
        metadata: Optional metadata about the document
        
    Returns:
        Status message
    """
    try:
        # Save the uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            file_path = tmp_file.name
        
        # Process the document
        result = agent.learn_from_document(file_path, metadata)
        
        # Clean up
        os.unlink(file_path)
        
        # Update KB stats
        st.session_state.kb_stats = get_kb_stats(agent)
        
        return result
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        return f"Error: {str(e)}"


def handle_url_addition(agent: ResearchAgent, url: str, max_depth: int = 1, metadata: Dict[str, Any] = None) -> str:
    """
    Add content from a URL to the knowledge base.
    
    Args:
        agent: ResearchAgent instance
        url: URL to fetch
        max_depth: Maximum recursion depth
        metadata: Optional metadata
        
    Returns:
        Status message
    """
    try:
        result = agent.learn_from_url(url, max_depth=max_depth, metadata=metadata)
        
        # Update KB stats
        st.session_state.kb_stats = get_kb_stats(agent)
        
        return result
    except Exception as e:
        logger.error(f"Error adding URL: {str(e)}")
        return f"Error: {str(e)}"


def handle_config_upload(agent: ResearchAgent, uploaded_file) -> Dict[str, str]:
    """
    Process an uploaded configuration file and schedule tasks.
    
    Args:
        agent: ResearchAgent instance
        uploaded_file: The uploaded config file from Streamlit
        
    Returns:
        Dictionary of source IDs and status messages
    """
    try:
        # Save the uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            config_path = tmp_file.name
        
        # Initialize the scheduler if needed
        if st.session_state.scheduler is None:
            st.session_state.scheduler = SourceScheduler(research_agent=agent)
        
        scheduler = st.session_state.scheduler
        
        # Read the configuration file
        with open(config_path, 'r') as f:
            config = json.load(f)
            
        results = {}
        
        # Process sources
        if "sources" in config:
            for source in config["sources"]:
                source_id = source.get("source_id")
                source_type = source.get("source_type")
                
                if not source_id or not source_type:
                    results[f"error_{len(results)}"] = "Missing source_id or source_type in source definition"
                    continue
                
                if source_type == "url":
                    url = source.get("url")
                    if not url:
                        results[source_id] = "Missing url in url source"
                        continue
                    
                    interval_minutes = source.get("interval_minutes")
                    cron_expression = source.get("cron_expression")
                    max_depth = source.get("max_depth", 1)
                    metadata = source.get("metadata", {})
                    
                    result = scheduler.add_url_source(
                        source_id=source_id,
                        url=url,
                        interval_minutes=interval_minutes,
                        cron_expression=cron_expression,
                        max_depth=max_depth,
                        metadata=metadata
                    )
                    results[source_id] = result
                elif source_type == "arxiv":
                    search_query = source.get("search_query")
                    if not search_query:
                        results[source_id] = "Missing search_query in arxiv source"
                        continue
                    
                    interval_hours = source.get("interval_hours")
                    cron_expression = source.get("cron_expression")
                    categories = source.get("categories")
                    max_results = source.get("max_results", 10)
                    metadata = source.get("metadata", {})
                    
                    result = scheduler.add_arxiv_source(
                        source_id=source_id,
                        search_query=search_query,
                        interval_hours=interval_hours,
                        cron_expression=cron_expression,
                        categories=categories,
                        max_results=max_results,
                        metadata=metadata
                    )
                    results[source_id] = result
                else:
                    results[source_id] = f"Unsupported source type: {source_type}"
        
        # Process scheduled tasks
        if "scheduled_tasks" in config:
            for task in config["scheduled_tasks"]:
                task_id = task.get("task_id")
                task_type = task.get("task_type")
                
                if not task_id or not task_type:
                    results[f"error_task_{len(results)}"] = "Missing task_id or task_type in scheduled task definition"
                    continue
                    
                if task_type == "prompt_on_url":
                    url = task.get("url")
                    prompt_template = task.get("prompt_template")
                    
                    if not url or not prompt_template:
                        results[task_id] = "Missing url or prompt_template"
                        continue
                        
                    interval_minutes = task.get("interval_minutes")
                    cron_expression = task.get("cron_expression")
                    output_action = task.get("output_action", {})
                    metadata = task.get("metadata", {})
                    
                    result = scheduler.add_prompt_on_url_task(
                        task_id=task_id,
                        url=url,
                        prompt_template=prompt_template,
                        interval_minutes=interval_minutes,
                        cron_expression=cron_expression,
                        output_action=output_action,
                        metadata=metadata
                    )
                    results[task_id] = result
                else:
                    results[task_id] = f"Unsupported task type: {task_type}"
        
        # Clean up
        os.unlink(config_path)
        
        return results
    except Exception as e:
        logger.error(f"Error scheduling from config: {str(e)}")
        return {"error": f"Error: {str(e)}"}


def display_chat_interface():
    """Display the chat interface for querying the agent."""
    st.header("🔍 Query the Agent", divider="blue")
    
    # Display chat history
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.chat_message("user").write(message["content"])
        else:
            st.chat_message("assistant").write(message["content"])
    
    # Input for new query
    if query := st.chat_input("Ask a question..."):
        st.chat_message("user").write(query)
        with st.spinner("Thinking..."):
            response = handle_query(st.session_state.agent, query)
        st.chat_message("assistant").write(response["output"])


def display_document_uploader():
    """Display the document uploader interface."""
    st.header("📄 Add Document", divider="blue")
    
    uploaded_file = st.file_uploader(
        "Upload a document (PDF, DOCX, TXT, MD, CSV)",
        type=["pdf", "docx", "txt", "md", "csv"]
    )
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        metadata_str = st.text_area(
            "Metadata (Optional JSON)",
            value='{"source": "manual upload", "topic": "general"}'
        )
    
    with col2:
        if st.button("Add Document", disabled=uploaded_file is None):
            if uploaded_file is not None:
                try:
                    metadata = json.loads(metadata_str) if metadata_str else {}
                    with st.spinner("Processing document..."):
                        result = handle_document_upload(st.session_state.agent, uploaded_file, metadata)
                    st.success(result)
                except json.JSONDecodeError:
                    st.error("Invalid JSON format in metadata")
                except Exception as e:
                    st.error(f"Error: {str(e)}")


def display_url_adder():
    """Display the URL adder interface."""
    st.header("🔗 Add URL Content", divider="blue")
    
    url = st.text_input("URL to fetch and process")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        metadata_str = st.text_area(
            "Metadata (Optional JSON)",
            value='{"source": "web", "topic": "general"}',
            key="url_metadata"
        )
        max_depth = st.slider("Max recursion depth", min_value=0, max_value=3, value=1)
    
    with col2:
        if st.button("Add URL", disabled=not url):
            if url:
                try:
                    metadata = json.loads(metadata_str) if metadata_str else {}
                    with st.spinner(f"Fetching content from {url}..."):
                        result = handle_url_addition(
                            st.session_state.agent,
                            url,
                            max_depth=max_depth,
                            metadata=metadata
                        )
                    st.success(result)
                except json.JSONDecodeError:
                    st.error("Invalid JSON format in metadata")
                except Exception as e:
                    st.error(f"Error: {str(e)}")


def display_scheduler():
    """Display the scheduler interface."""
    st.header("⏰ Schedule Tasks", divider="blue")
    
    uploaded_file = st.file_uploader(
        "Upload a configuration JSON file",
        type=["json"]
    )
    
    if st.button("Schedule Tasks", disabled=uploaded_file is None):
        if uploaded_file is not None:
            with st.spinner("Processing configuration..."):
                results = handle_config_upload(st.session_state.agent, uploaded_file)
            
            st.subheader("Results")
            for source_id, result in results.items():
                st.write(f"**{source_id}**: {result}")
    
    # Show existing scheduled tasks if any
    if st.session_state.scheduler is not None:
        scheduler = st.session_state.scheduler
        
        st.subheader("Active Sources")
        sources = scheduler.get_all_sources()
        if sources:
            for source_id, source_info in sources.items():
                with st.expander(f"Source: {source_id} ({source_info.get('type', 'unknown')})"):
                    st.json(source_info)
        else:
            st.info("No active sources")
        
        st.subheader("Active Tasks")
        tasks = scheduler.scheduled_tasks
        if tasks:
            for task_id, task_info in tasks.items():
                with st.expander(f"Task: {task_id}"):
                    # Extract displayable information
                    display_info = {
                        "url": task_info.get("url", ""),
                        "interval_minutes": task_info.get("interval_minutes", ""),
                        "cron_expression": task_info.get("cron_expression", ""),
                        "prompt_template": task_info.get("prompt_template", "").split("\n")[0] + "..." if task_info.get("prompt_template", "") else "",
                        "output_action": task_info.get("output_action", {}).get("type", "")
                    }
                    st.json(display_info)
        else:
            st.info("No active tasks")


def display_kb_viewer():
    """Display a simple knowledge base viewer."""
    st.header("📚 Knowledge Base", divider="blue")
    
    # Display KB stats
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Documents", st.session_state.kb_stats["document_count"])
    with col2:
        st.metric("Chunks", st.session_state.kb_stats["chunk_count"])
    
    if st.button("Refresh Stats"):
        with st.spinner("Updating stats..."):
            st.session_state.kb_stats = get_kb_stats(st.session_state.agent)
        st.success("Stats updated!")
    
    # Simple search interface
    st.subheader("Search Knowledge Base")
    search_query = st.text_input("Search term")
    
    if st.button("Search", disabled=not search_query):
        if search_query:
            with st.spinner("Searching..."):
                try:
                    kb = st.session_state.agent.retriever.knowledge_base
                    docs = kb.similarity_search(search_query, k=5)
                    
                    if docs:
                        for i, doc in enumerate(docs):
                            with st.expander(f"Result {i+1} - {doc.metadata.get('title', 'Unknown')}"):
                                st.write(f"**Source**: {doc.metadata.get('document_id', 'Unknown')}")
                                st.write(f"**Content Hash**: {doc.metadata.get('content_hash', 'Unknown')}")
                                st.write("**Content**:")
                                st.write(doc.page_content)
                    else:
                        st.info("No matching documents found.")
                except Exception as e:
                    st.error(f"Error searching knowledge base: {str(e)}")


def main():
    """Main function to run the Streamlit app."""
    st.title("🧠 AI Research Agent")
    
    # Initialize the agent if needed
    if st.session_state.agent is None:
        with st.spinner("Initializing agent..."):
            st.session_state.agent = setup_agent()
            st.session_state.kb_stats = get_kb_stats(st.session_state.agent)
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select a page",
        ["Chat", "Add Document", "Add URL", "Schedule Tasks", "Knowledge Base"]
    )
    
    # Display agent info in sidebar
    agent = st.session_state.agent
    st.sidebar.divider()
    st.sidebar.subheader("Agent Info")
    st.sidebar.info(
        f"**Agent**: {agent.agent_name}\n\n"
        f"**LLM Provider**: {agent.llm_provider}\n\n"
        f"**Model**: {agent.model_name}\n\n"
        f"**Temperature**: {agent.temperature}"
    )
    
    # Display the selected page
    if page == "Chat":
        display_chat_interface()
    elif page == "Add Document":
        display_document_uploader()
    elif page == "Add URL":
        display_url_adder()
    elif page == "Schedule Tasks":
        display_scheduler()
    elif page == "Knowledge Base":
        display_kb_viewer()


if __name__ == "__main__":
    main()
