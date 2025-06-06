"""
Web UI controller for the AI Research Agent.
This module provides FastAPI routes for HTML-based web interface.
"""

import os
import json
import logging
import re
from typing import List, Dict, Optional, Any

from fastapi import APIRouter, Request, Depends, Form, UploadFile, File, Cookie, Response, status
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse

# Import our memory log handler
from app.ui.log_handler import memory_log_handler, setup_memory_logging

# Set up memory logging
setup_memory_logging()

from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from app.api import get_agent, get_scheduler
from app.core.agent import ResearchAgent
from app.schedulers.source_scheduler import SourceScheduler

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/ui")

# Initialize Jinja2 templates
templates_path = os.path.join(os.path.dirname(__file__), "templates")
templates = Jinja2Templates(directory=templates_path)

# Initialize session state
chat_histories = {}  # Simple in-memory cache of chat histories

# Function to get sanitized environment variables for display
def get_global_settings():
    """Read environment variables and sanitize sensitive data for display"""
    # Settings to display (add more as needed)
    display_settings = {
        "LLM_PROVIDER": os.getenv("LLM_PROVIDER", "openai"),
        "TEMPERATURE": os.getenv("TEMPERATURE", "0.0"),
        "KNOWLEDGE_BASE_DIR": os.getenv("KNOWLEDGE_BASE_DIR", "./knowledge_base"),
        "MAX_CRAWL_URLS": os.getenv("MAX_CRAWL_URLS", "100"),
        "MAX_CRAWL_URLS_PER_PAGE": os.getenv("MAX_CRAWL_URLS_PER_PAGE", "50"),
    }
    
    # Add model name based on provider
    if display_settings["LLM_PROVIDER"].lower() == "ollama":
        display_settings["MODEL_NAME"] = os.getenv("OLLAMA_MODEL_NAME", "llama3:latest")
        display_settings["EMBEDDING_MODEL"] = os.getenv("OLLAMA_EMBEDDING_MODEL_NAME", "nomic-embed-text")
    else:
        display_settings["MODEL_NAME"] = os.getenv("MODEL_NAME", "")
        # For OpenAI, don't display the actual API key, just whether it's set
        if os.getenv("OPENAI_API_KEY"):
            display_settings["API_KEY_STATUS"] = "Set ✓"
        else:
            display_settings["API_KEY_STATUS"] = "Not Set ✗"
    
    return display_settings

# UI Routes
@router.get("/", response_class=HTMLResponse)
async def index(request: Request, agent: ResearchAgent = Depends(get_agent)):
    """Render the chat interface"""
    session_id = request.cookies.get("session_id", "default")
    
    # Get or initialize chat history
    chat_history = chat_histories.get(session_id, [])
    
    # Determine the correct model name to display based on provider
    llm_provider = os.getenv("LLM_PROVIDER", "openai").lower()
    if llm_provider == "ollama":
        model_name = os.getenv("OLLAMA_MODEL_NAME", "llama3:latest")
    else:
        model_name = agent.model_name
    
    return templates.TemplateResponse("index.html", {
        "request": request,
        "active_page": "chat",
        "chat_history": chat_history,
        "agent_name": agent.agent_name,
        "model_name": model_name,
        "llm_provider": llm_provider,
        "temperature": os.getenv("TEMPERATURE", "0.0"),
        "global_settings": get_global_settings()
    })

@router.post("/chat", response_class=HTMLResponse)
async def chat(
    request: Request,
    query: str = Form(...),
    use_kb: Optional[str] = Form(None),
    agent: ResearchAgent = Depends(get_agent)
):
    """Process a chat query and redirect back to the chat interface"""
    session_id = request.cookies.get("session_id", "default")
    
    # Get or initialize chat history
    chat_history = chat_histories.get(session_id, [])
    
    # Add user message to history
    chat_history.append({"role": "user", "content": query})
    
    try:
        # Determine if knowledge base should be used
        using_kb = use_kb == "true"
        
        # Add KB usage info to query metadata for logging
        kb_mode = "with KB" if using_kb else "without KB (LLM-only)"
        logger.info(f"Processing query: '{query}' {kb_mode}")
        
        # Get response from agent with KB toggle parameter
        response = agent.run(query, chat_history, use_knowledge_base=using_kb)
        
        # Extract only the 'output' field from the response JSON if it exists
        if isinstance(response, dict) and 'output' in response:
            agent_response = response['output']
        else:
            # Fallback to using the entire response if 'output' field is not found
            agent_response = str(response)
            
        # Prepend a notice about KB usage mode
        kb_notice = f"*[Response generated {'with' if using_kb else 'WITHOUT'} Knowledge Base]*\n\n"
        agent_response = kb_notice + agent_response
        
        # Add agent response to history
        chat_history.append({"role": "assistant", "content": agent_response})
        
        # Update chat history in session
        chat_histories[session_id] = chat_history
        
    except Exception as e:
        logger.error(f"Error in chat: {str(e)}")
        # Add error message to history
        chat_history.append({
            "role": "assistant", 
            "content": f"Error: {str(e)}"
        })
        chat_histories[session_id] = chat_history
    
    # Redirect back to the chat interface
    return RedirectResponse(url="/ui/", status_code=303)

@router.post("/clear-chat")
async def clear_chat(request: Request):
    """Clear the chat history for the current session"""
    session_id = request.cookies.get("session_id", "default")
    
    # Clear the chat history for this session
    if session_id in chat_histories:
        chat_histories[session_id] = []
        logger.info(f"Cleared chat history for session {session_id}")
    
    return {"status": "success", "message": "Chat history cleared"}

@router.get("/add-document", response_class=HTMLResponse)
async def add_document_form(
    request: Request,
    message: Optional[str] = None,
    message_type: str = "info",
    agent: ResearchAgent = Depends(get_agent)
):
    """Render the document upload form"""
    # Determine the correct model name to display based on provider
    llm_provider = os.getenv("LLM_PROVIDER", "openai").lower()
    if llm_provider == "ollama":
        model_name = os.getenv("OLLAMA_MODEL_NAME", "llama3:latest")
    else:
        model_name = agent.model_name
    
    return templates.TemplateResponse("add_document.html", {
        "request": request,
        "active_page": "add-document",
        "message": message,
        "message_type": message_type,
        "agent_name": agent.agent_name,
        "model_name": model_name,
        "llm_provider": llm_provider,
        "temperature": os.getenv("TEMPERATURE", "0.0"),
        "global_settings": get_global_settings()
    })

@router.post("/add-document", response_class=HTMLResponse)
async def process_document(
    request: Request,
    file: UploadFile = File(...),
    metadata: Optional[str] = Form(None),
    agent: ResearchAgent = Depends(get_agent)
):
    """Process an uploaded document"""
    try:
        # Save the uploaded file temporarily
        temp_path = f"/tmp/{file.filename}"
        with open(temp_path, "wb") as f:
            content = await file.read()
            f.write(content)
            
        # Parse metadata if provided
        meta_dict = None
        if metadata and metadata.strip():
            try:
                meta_dict = json.loads(metadata)
            except json.JSONDecodeError:
                return templates.TemplateResponse("add_document.html", {
                    "request": request,
                    "active_page": "add-document",
                    "message": "Invalid JSON in metadata field",
                    "message_type": "danger",
                    "agent_name": agent.agent_name,
                    "model_name": agent.model_name,
                    "llm_provider": os.getenv("LLM_PROVIDER", "Unknown"),
                    "temperature": os.getenv("TEMPERATURE", "0.0"),
                    "global_settings": get_global_settings()
                })
        
        # Learn from the document
        result = agent.learn_from_document(temp_path, meta_dict)
        
        # Clean up
        os.remove(temp_path)
        
        # Determine the correct model name to display based on provider
        llm_provider = os.getenv("LLM_PROVIDER", "openai").lower()
        if llm_provider == "ollama":
            model_name = os.getenv("OLLAMA_MODEL_NAME", "llama3:latest")
        else:
            model_name = agent.model_name
        
        return templates.TemplateResponse("add_document.html", {
            "request": request,
            "active_page": "add-document",
            "message": f"Document processed successfully: {result}",
            "message_type": "success",
            "agent_name": agent.agent_name,
            "model_name": model_name,
            "llm_provider": llm_provider,
            "temperature": os.getenv("TEMPERATURE", "0.0"),
            "global_settings": get_global_settings()
        })
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        # Determine the correct model name to display based on provider
        llm_provider = os.getenv("LLM_PROVIDER", "openai").lower()
        if llm_provider == "ollama":
            model_name = os.getenv("OLLAMA_MODEL_NAME", "llama3:latest")
        else:
            model_name = agent.model_name
        
        return templates.TemplateResponse("add_document.html", {
            "request": request,
            "active_page": "add-document",
            "message": f"Error processing document: {str(e)}",
            "message_type": "danger",
            "agent_name": agent.agent_name,
            "model_name": model_name,
            "llm_provider": llm_provider,
            "temperature": os.getenv("TEMPERATURE", "0.0"),
            "global_settings": get_global_settings()
        })

@router.get("/add-url", response_class=HTMLResponse)
async def add_url_form(
    request: Request,
    message: Optional[str] = None,
    message_type: str = "info",
    agent: ResearchAgent = Depends(get_agent)
):
    """Render the URL form"""
    # Determine the correct model name to display based on provider
    llm_provider = os.getenv("LLM_PROVIDER", "openai").lower()
    if llm_provider == "ollama":
        model_name = os.getenv("OLLAMA_MODEL_NAME", "llama3:latest")
    else:
        model_name = agent.model_name
    
    return templates.TemplateResponse("add_url.html", {
        "request": request,
        "active_page": "add-url",
        "message": message,
        "message_type": message_type,
        "agent_name": agent.agent_name,
        "model_name": model_name,
        "llm_provider": llm_provider,
        "temperature": os.getenv("TEMPERATURE", "0.0"),
        "global_settings": get_global_settings()
    })

@router.post("/add-url", response_class=HTMLResponse)
async def process_url(
    request: Request,
    url: str = Form(...),
    max_depth: int = Form(0),
    max_urls: int = Form(100),
    max_urls_per_page: int = Form(50),
    excluded_domains: Optional[str] = Form(None),
    metadata: Optional[str] = Form(None),
    use_js: Optional[bool] = Form(False),
    agent: ResearchAgent = Depends(get_agent)
):
    """Process a submitted URL"""
    try:
        # Log the URL processing request
        logger.info(f"Processing URL: {url} with max_depth={max_depth}")
        
        # Parse metadata if provided
        meta_dict = None
        if metadata and metadata.strip():
            try:
                meta_dict = json.loads(metadata)
                logger.info(f"Using custom metadata: {meta_dict}")
            except json.JSONDecodeError:
                logger.warning(f"Invalid JSON in metadata: {metadata}")
                
                # Determine the correct model name to display based on provider
                llm_provider = os.getenv("LLM_PROVIDER", "openai").lower()
                if llm_provider == "ollama":
                    model_name = os.getenv("OLLAMA_MODEL_NAME", "llama3:latest")
                else:
                    model_name = agent.model_name
                
                return templates.TemplateResponse("add_url.html", {
                    "request": request,
                    "active_page": "add-url",
                    "message": "Invalid JSON in metadata field",
                    "message_type": "danger",
                    "agent_name": agent.agent_name,
                    "model_name": model_name,
                    "llm_provider": llm_provider,
                    "temperature": os.getenv("TEMPERATURE", "0.0")
                })
        
        # Initialize metadata dictionary if not provided
        if meta_dict is None:
            meta_dict = {}
            
        # Include max_depth in the metadata if specified
        if max_depth > 0:
            meta_dict["max_depth"] = max_depth
            logger.info(f"Setting max_depth={max_depth} in metadata")
            
        # Include max_urls and max_urls_per_page in the metadata
        meta_dict["max_urls"] = max_urls
        meta_dict["max_urls_per_page"] = max_urls_per_page
        logger.info(f"Setting max_urls={max_urls} and max_urls_per_page={max_urls_per_page} in metadata")
            
        # Include use_js flag in metadata if checked
        if use_js:
            meta_dict["use_js"] = True
            logger.info(f"Enabling JavaScript rendering for URL: {url}")
            
        # Process excluded domains if provided
        excluded_domains_list = None
        if excluded_domains and excluded_domains.strip():
            excluded_domains_list = [domain.strip() for domain in excluded_domains.split(",") if domain.strip()]
            if excluded_domains_list:
                logger.info(f"Excluding domains: {', '.join(excluded_domains_list)}")
                # Store in metadata for display purposes - as a string, not a list
                # ChromaDB only accepts primitive types, not lists
                meta_dict["excluded_domains"] = ",".join(excluded_domains_list)
        
        # Log the start of URL processing
        if max_depth > 0:
            logger.info(f"Starting recursive URL processing for {url} with depth {max_depth}")
            logger.info(f"Check the logs page to monitor processing progress")
            
            # For recursive processing, call the specific recursive method
            result = agent.retriever.add_recursive_url_content(
                url, 
                max_depth=max_depth, 
                metadata=meta_dict,
                excluded_domains=excluded_domains_list
            )
        else:
            logger.info(f"Processing single URL: {url}")
            # For single URL, use the standard method
            result = agent.learn_from_url(url, meta_dict)
        
        logger.info(f"URL processing complete: {url}")

        
        # Determine the correct model name to display based on provider
        llm_provider = os.getenv("LLM_PROVIDER", "openai").lower()
        if llm_provider == "ollama":
            model_name = os.getenv("OLLAMA_MODEL_NAME", "llama3:latest")
        else:
            model_name = agent.model_name
            
        return templates.TemplateResponse("add_url.html", {
            "request": request,
            "active_page": "add-url",
            "message": f"URL processed successfully. {result} View logs for details.",
            "message_type": "success",
            "agent_name": agent.agent_name,
            "model_name": model_name,
            "llm_provider": llm_provider,
            "temperature": os.getenv("TEMPERATURE", "0.0"),
            "global_settings": get_global_settings()
        })
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error processing URL: {error_msg}")
        
        # Determine the correct model name to display based on provider
        llm_provider = os.getenv("LLM_PROVIDER", "openai").lower()
        if llm_provider == "ollama":
            model_name = os.getenv("OLLAMA_MODEL_NAME", "llama3:latest")
        else:
            model_name = agent.model_name
            
        return templates.TemplateResponse("add_url.html", {
            "request": request,
            "active_page": "add-url",
            "message": f"Error processing URL: {error_msg}",
            "message_type": "danger",
            "agent_name": agent.agent_name,
            "model_name": model_name,
            "llm_provider": llm_provider,
            "temperature": os.getenv("TEMPERATURE", "0.0"),
            "global_settings": get_global_settings()
        })

@router.get("/knowledge-base", response_class=HTMLResponse)
async def knowledge_base(
    request: Request,
    search: Optional[str] = None,
    refresh: Optional[int] = None,
    document_id: Optional[str] = None,
    agent: ResearchAgent = Depends(get_agent)
):
    """Render the knowledge base page"""
    # Get KB stats
    kb_stats = agent.knowledge_base.get_collection_stats()
    
    # Search results
    search_results = []
    document_chunks = []
    all_documents = {}
    
    # If document_id is provided, get chunks for that document
    if document_id:
        try:
            document_chunks = agent.knowledge_base.get_document_chunks(document_id)
            logger.info(f"Retrieved {len(document_chunks)} chunks for document: {document_id}")
        except Exception as e:
            logger.error(f"Error retrieving document chunks: {str(e)}")
    # Otherwise get all document IDs for browsing
    else:
        try:
            all_documents = agent.knowledge_base.get_all_documents()
            logger.info(f"Retrieved {len(all_documents)} documents from knowledge base")
        except Exception as e:
            logger.error(f"Error retrieving document list: {str(e)}")
    
    # If search is provided, perform search
    if search:
        try:
            # Use similarity_search instead of search
            docs = agent.knowledge_base.similarity_search(search, k=10)
            # Convert Document objects to dictionaries for display
            search_results = [{
                'content': doc.page_content,
                'metadata': doc.metadata
            } for doc in docs]
        except Exception as e:
            logger.error(f"Search error: {str(e)}")
    
    # Determine the correct model name to display based on provider
    llm_provider = os.getenv("LLM_PROVIDER", "openai").lower()
    if llm_provider == "ollama":
        model_name = os.getenv("OLLAMA_MODEL_NAME", "llama3:latest")
    else:
        model_name = agent.model_name
    
    return templates.TemplateResponse("knowledge_base.html", {
        "request": request,
        "active_page": "kb",
        "kb_stats": kb_stats,
        "search_term": search,
        "search_results": search_results,
        "document_id": document_id,
        "document_chunks": document_chunks,
        "all_documents": all_documents,
        "agent_name": agent.agent_name,
        "model_name": model_name,
        "llm_provider": llm_provider,
        "temperature": os.getenv("TEMPERATURE", "0.0"),
        "global_settings": get_global_settings()
    })

@router.get("/scheduler", response_class=HTMLResponse)
async def scheduler(
    request: Request,
    message: Optional[str] = None,
    message_type: str = "info",
    agent: ResearchAgent = Depends(get_agent),
    scheduler: SourceScheduler = Depends(get_scheduler)
):
    """Render the scheduler page"""
    # Get sources and tasks
    sources = scheduler.get_all_sources()
    
    # Get tasks (if scheduler supports it)
    tasks = {}
    if hasattr(scheduler, 'get_all_scheduled_tasks') and callable(getattr(scheduler, 'get_all_scheduled_tasks')):
        tasks = scheduler.get_all_scheduled_tasks()
        logger.info(f"Found {len(tasks)} scheduled tasks: {list(tasks.keys())}")
    elif hasattr(scheduler, 'get_all_tasks') and callable(getattr(scheduler, 'get_all_tasks')):
        tasks = scheduler.get_all_tasks()
    elif hasattr(scheduler, 'prompt_on_url_tasks') and hasattr(scheduler.prompt_on_url_tasks, 'items'):
        tasks = scheduler.prompt_on_url_tasks
    else:
        # Direct access to scheduled_tasks attribute as fallback
        if hasattr(scheduler, 'scheduled_tasks'):
            tasks = scheduler.scheduled_tasks
            logger.info(f"Using fallback scheduler.scheduled_tasks with {len(tasks)} tasks")
    
    # Determine the correct model name to display based on provider
    llm_provider = os.getenv("LLM_PROVIDER", "openai").lower()
    if llm_provider == "ollama":
        model_name = os.getenv("OLLAMA_MODEL_NAME", "llama3:latest")
    else:
        model_name = agent.model_name
    
    return templates.TemplateResponse("scheduler.html", {
        "request": request,
        "active_page": "scheduler",
        "sources": sources,
        "tasks": tasks,
        "message": message,
        "message_type": message_type,
        "agent_name": agent.agent_name,
        "model_name": model_name,
        "llm_provider": llm_provider,
        "temperature": os.getenv("TEMPERATURE", "0.0"),
        "global_settings": get_global_settings()
    })

@router.post("/scheduler/source/remove")
async def remove_scheduler_source(
    request: Request,
    source_id: str = Form(...),
    scheduler: SourceScheduler = Depends(get_scheduler)
):
    """Remove a scheduled source"""
    try:
        scheduler.remove_source(source_id)
        return RedirectResponse(
            url=f"/ui/scheduler?message=Source {source_id} removed successfully&message_type=success",
            status_code=status.HTTP_303_SEE_OTHER
        )
    except Exception as e:
        logger.error(f"Error removing source {source_id}: {e}", exc_info=True)
        return RedirectResponse(
            url=f"/ui/scheduler?message=Error removing source: {str(e)}&message_type=danger",
            status_code=status.HTTP_303_SEE_OTHER
        )


@router.post("/scheduler/task/remove")
async def remove_scheduled_task(
    request: Request,
    task_id: str = Form(...),
    scheduler: SourceScheduler = Depends(get_scheduler)
):
    """Remove a scheduled task"""
    try:
        # Check if scheduler has the method to remove scheduled tasks
        if hasattr(scheduler, 'remove_scheduled_task') and callable(getattr(scheduler, 'remove_scheduled_task')):
            scheduler.remove_scheduled_task(task_id)
            logger.info(f"Removed scheduled task {task_id}")
        else:
            # Fallback approach - access the job by ID and remove it directly
            if hasattr(scheduler, 'scheduled_tasks') and task_id in scheduler.scheduled_tasks:
                # Get the job ID and remove from the APScheduler
                job_id = scheduler.scheduled_tasks[task_id].get('job_id')
                if job_id and scheduler.scheduler.get_job(job_id):
                    scheduler.scheduler.remove_job(job_id)
                # Remove from the tasks dictionary
                del scheduler.scheduled_tasks[task_id]
                logger.info(f"Removed scheduled task {task_id} using fallback method")
            else:
                raise ValueError(f"Task {task_id} not found in scheduled tasks")
                
        return RedirectResponse(
            url=f"/ui/scheduler?message=Task {task_id} removed successfully&message_type=success",
            status_code=status.HTTP_303_SEE_OTHER
        )
    except Exception as e:
        logger.error(f"Error removing task {task_id}: {e}", exc_info=True)
        return RedirectResponse(
            url=f"/ui/scheduler?message=Error removing task: {str(e)}&message_type=danger",
            status_code=status.HTTP_303_SEE_OTHER
        )

@router.post("/scheduler/upload")
async def upload_config(
    request: Request,
    config_file: UploadFile = File(...),
    scheduler: SourceScheduler = Depends(get_scheduler)
):
    """Upload a scheduler configuration file"""
    try:
        # Read the config file
        content = await config_file.read()
        config = json.loads(content)
        
        # Process scheduled tasks
        if "scheduled_tasks" in config:
            added_count = 0
            for task in config["scheduled_tasks"]:
                try:
                    # Check for either task_type or type (for backward compatibility)
                    task_type = task.get("task_type") or task.get("type")
                    if task_type == "prompt_on_url":
                        scheduler.add_prompt_on_url_task(
                            task_id=task.get("task_id") or task.get("id"),
                            url=task.get("url"),
                            prompt_template=task.get("prompt_template"),
                            interval_minutes=task.get("interval_minutes", 60),
                            output_action=task.get("output_action"),
                            metadata=task.get("metadata")
                        )
                        added_count += 1
                        # Log successful addition of task
                        logger.info(f"Added scheduled task: {task.get('task_id') or task.get('id')}")
                except Exception as task_error:
                    logger.error(f"Error adding task: {str(task_error)}", exc_info=True)
            
            return RedirectResponse(
                url=f"/ui/scheduler?message=Added {added_count} tasks from configuration&message_type=success",
                status_code=303
            )
        else:
            return RedirectResponse(
                url="/ui/scheduler?message=No scheduled_tasks found in configuration&message_type=warning",
                status_code=303
            )
    except json.JSONDecodeError:
        return RedirectResponse(
            url="/ui/scheduler?message=Invalid JSON configuration file&message_type=danger",
            status_code=303
        )
    except Exception as e:
        logger.error(f"Error uploading configuration: {str(e)}")
        return RedirectResponse(
            url=f"/ui/scheduler?message=Error uploading configuration: {str(e)}&message_type=danger",
            status_code=303
        )

# ==== Log API Endpoints ====
@router.get("/api/logs")
async def get_logs(request: Request, since_index: int = 0):
    """Get logs since the given index."""
    # Get the session ID from the request's session cookie
    session_id = request.cookies.get('session_id', 'default')
    
    # Get logs since the given index
    logs_data = memory_log_handler.get_logs(session_id, since_index)
    
    # Return the logs and next index
    return logs_data

@router.get("/logs", response_class=HTMLResponse)
async def log_viewer(request: Request, agent: ResearchAgent = Depends(get_agent)):
    """Render the log viewer page."""
    # Determine the correct model name to display based on provider
    llm_provider = os.getenv("LLM_PROVIDER", "openai").lower()
    if llm_provider == "ollama":
        model_name = os.getenv("OLLAMA_MODEL_NAME", "llama3:latest")
    else:
        model_name = agent.model_name
    
    return templates.TemplateResponse("logs.html", {
        "request": request,
        "active_page": "logs",
        "agent_name": agent.agent_name,
        "model_name": model_name,
        "llm_provider": llm_provider,
        "temperature": os.getenv("TEMPERATURE", "0.0"),
        "global_settings": get_global_settings()
    })
