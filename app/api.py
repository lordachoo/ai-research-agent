"""
API implementation for the AI Research Agent.
This module provides a FastAPI interface for interacting with the agent.
"""

import os
import json
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

from app.core.agent import ResearchAgent
from app.core.knowledge_base import KnowledgeBase
from app.schedulers.source_scheduler import SourceScheduler


# Define API models
class QueryRequest(BaseModel):
    query: str = Field(..., description="The query to send to the agent")
    chat_history: Optional[List[Dict[str, str]]] = Field(default=None, description="Optional chat history")


class SourceConfig(BaseModel):
    source_id: str = Field(..., description="Unique identifier for the source")
    source_type: str = Field(..., description="Type of source (url, arxiv, custom)")
    url: Optional[str] = Field(default=None, description="URL for URL sources")
    search_query: Optional[str] = Field(default=None, description="Search query for arXiv sources")
    categories: Optional[List[str]] = Field(default=None, description="Categories for arXiv sources")
    interval_minutes: Optional[int] = Field(default=60, description="Interval in minutes between checks")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Optional metadata about the source")
    max_depth: Optional[int] = Field(default=0, description="Depth for recursive URL fetching")
    max_results: Optional[int] = Field(default=10, description="Maximum number of results for arXiv sources")


# Global variables to store agent and scheduler instances
research_agent = None
source_scheduler = None


# Initialize FastAPI app
app = FastAPI(
    title="AI Research Agent API",
    description="API for interacting with the AI Research Agent",
    version="1.0.0",
)


# Dependency to get the agent
def get_agent():
    global research_agent
    if research_agent is None:
        # Load environment variables
        import dotenv
        dotenv.load_dotenv()
        
        # Initialize knowledge base
        kb_dir = os.getenv("KNOWLEDGE_BASE_DIR", "./knowledge_base")
        kb = KnowledgeBase(persist_directory=kb_dir)
        
        # Initialize agent
        model_name = os.getenv("MODEL_NAME", "gpt-3.5-turbo-0125")
        research_agent = ResearchAgent(
            agent_name=os.getenv("AGENT_NAME", "Research Assistant"),
            knowledge_base=kb,
            model_name=model_name,
            temperature=float(os.getenv("TEMPERATURE", "0.1")),
        )
    return research_agent


# Dependency to get the scheduler
def get_scheduler():
    global source_scheduler
    if source_scheduler is None:
        agent = get_agent()
        source_scheduler = SourceScheduler(research_agent=agent)
    return source_scheduler


# Define API routes
@app.get("/")
async def root():
    return {"message": "AI Research Agent API"}


@app.post("/query")
async def query_agent(request: QueryRequest, agent: ResearchAgent = Depends(get_agent)):
    try:
        response = agent.run(request.query, request.chat_history)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/learn/document")
async def learn_from_document(
    file: UploadFile = File(...),
    metadata: Optional[str] = Form(default=None),
    agent: ResearchAgent = Depends(get_agent)
):
    try:
        # Save the uploaded file temporarily
        temp_path = f"/tmp/{file.filename}"
        with open(temp_path, "wb") as f:
            f.write(await file.read())
            
        # Parse metadata if provided
        meta_dict = json.loads(metadata) if metadata else None
        
        # Learn from the document
        result = agent.learn_from_document(temp_path, meta_dict)
        
        # Clean up
        os.remove(temp_path)
        
        return {"message": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/learn/url")
async def learn_from_url(
    url: str = Form(...),
    metadata: Optional[str] = Form(default=None),
    max_depth: int = Form(default=0),
    agent: ResearchAgent = Depends(get_agent)
):
    try:
        # Parse metadata if provided
        meta_dict = json.loads(metadata) if metadata else None
        
        # Learn from the URL
        if max_depth > 0:
            result = agent.retriever.add_recursive_url_content(url, max_depth=max_depth, metadata=meta_dict)
        else:
            result = agent.learn_from_url(url, meta_dict)
            
        return {"message": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/learn/text")
async def learn_from_text(
    text: str = Form(...),
    metadata: Optional[str] = Form(default=None),
    agent: ResearchAgent = Depends(get_agent)
):
    try:
        # Parse metadata if provided
        meta_dict = json.loads(metadata) if metadata else None
        
        # Learn from the text
        result = agent.retriever.add_text(text, meta_dict)
        
        return {"message": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/sources")
async def add_source(
    config: SourceConfig,
    scheduler: SourceScheduler = Depends(get_scheduler)
):
    try:
        if config.source_type == "url":
            if not config.url:
                raise HTTPException(status_code=400, detail="URL is required for URL sources")
                
            result = scheduler.add_url_source(
                source_id=config.source_id,
                url=config.url,
                interval_minutes=config.interval_minutes or 60,
                metadata=config.metadata,
                max_depth=config.max_depth or 0
            )
        elif config.source_type == "arxiv":
            if not config.search_query:
                raise HTTPException(status_code=400, detail="Search query is required for arXiv sources")
                
            result = scheduler.add_arxiv_source(
                source_id=config.source_id,
                search_query=config.search_query,
                categories=config.categories,
                interval_hours=(config.interval_minutes or 60) // 60,
                max_results=config.max_results or 10,
                metadata=config.metadata
            )
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported source type: {config.source_type}")
            
        return {"message": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/sources/{source_id}")
async def remove_source(
    source_id: str,
    scheduler: SourceScheduler = Depends(get_scheduler)
):
    try:
        result = scheduler.remove_source(source_id)
        return {"message": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sources")
async def get_sources(
    source_id: Optional[str] = Query(default=None),
    scheduler: SourceScheduler = Depends(get_scheduler)
):
    try:
        if source_id:
            source = scheduler.get_source(source_id)
            if not source:
                raise HTTPException(status_code=404, detail=f"Source '{source_id}' not found")
            return source
        else:
            return scheduler.get_all_sources()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/sources/{source_id}/interval")
async def update_source_interval(
    source_id: str,
    interval_minutes: int = Form(...),
    scheduler: SourceScheduler = Depends(get_scheduler)
):
    try:
        result = scheduler.update_source_interval(source_id, interval_minutes)
        return {"message": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/kb/stats")
async def get_kb_stats(agent: ResearchAgent = Depends(get_agent)):
    try:
        stats = agent.knowledge_base.get_collection_stats()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/kb")
async def reset_kb(agent: ResearchAgent = Depends(get_agent)):
    try:
        result = agent.knowledge_base.delete_collection()
        return {"message": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Run the API if this file is executed directly
if __name__ == "__main__":
    uvicorn.run("app.api:app", host="0.0.0.0", port=8000, reload=True)
