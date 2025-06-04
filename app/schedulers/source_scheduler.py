"""
Scheduler implementation for the AI Research Agent.
This module handles scheduling periodic checks of sources for new information.
"""

import os
import time
import datetime
import logging
from typing import Dict, Any, List, Optional, Callable
import threading
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.triggers.cron import CronTrigger
from apscheduler.jobstores.base import JobLookupError

from langchain_community.document_loaders import WebBaseLoader
from app.core.agent import ResearchAgent


class SourceScheduler:
    """Scheduler for periodically checking sources for new information."""

    def __init__(self, research_agent: ResearchAgent):
        """
        Initialize the Source Scheduler.

        Args:
            research_agent: ResearchAgent instance to use for learning from sources
        """
        self.research_agent = research_agent
        self.scheduler = BackgroundScheduler()
        self.scheduler.start()
        self.sources = {}
        self.logger = logging.getLogger(__name__)
        self.scheduled_tasks = {} # For the new task type
        
    def add_url_source(
        self, 
        source_id: str, 
        url: str, 
        interval_minutes: int = 60,
        metadata: Optional[Dict[str, Any]] = None,
        max_depth: int = 0
    ) -> str:
        """
        Add a URL source to check periodically.
        
        Args:
            source_id: Unique identifier for the source
            url: URL to check
            interval_minutes: Interval in minutes between checks
            metadata: Optional metadata about the source
            max_depth: Depth for recursive URL fetching (0 for single page)
            
        Returns:
            Status message
        """
        if source_id in self.sources:
            return f"Source with ID '{source_id}' already exists"
            
        # Create metadata if not provided
        if metadata is None:
            metadata = {}
            
        # Add source info to metadata
        metadata.update({
            "source_id": source_id,
            "source_type": "url",
            "url": url,
            "first_check": datetime.datetime.now().isoformat()
        })
        
        # Define the job function
        def check_url():
            try:
                metadata["last_check"] = datetime.datetime.now().isoformat()
                self.logger.info(f"Checking URL source: {source_id} - {url}")
                
                if max_depth > 0:
                    result = self.research_agent.retriever.add_recursive_url_content(
                        url, max_depth=max_depth, metadata=metadata
                    )
                else:
                    result = self.research_agent.learn_from_url(url, metadata=metadata)
                    
                self.logger.info(f"Result for {source_id}: {result}")
                return result
            except Exception as e:
                self.logger.error(f"Error checking URL source {source_id}: {str(e)}")
                return f"Error checking URL source {source_id}: {str(e)}"
        
        # Add the job to the scheduler
        job = self.scheduler.add_job(
            check_url,
            trigger=IntervalTrigger(minutes=interval_minutes),
            id=source_id,
            replace_existing=True,
            next_run_time=datetime.datetime.now()  # Run immediately first time
        )
        
        # Store the source
        self.sources[source_id] = {
            "id": source_id,
            "type": "url",
            "url": url,
            "interval_minutes": interval_minutes,
            "metadata": metadata,
            "max_depth": max_depth,
            "job_id": job.id,
            "added_at": datetime.datetime.now().isoformat()
        }
        
        return f"Successfully added URL source '{source_id}' with interval {interval_minutes} minutes"
        
    def add_arxiv_source(
        self, 
        source_id: str, 
        search_query: str,
        categories: Optional[List[str]] = None,
        interval_hours: int = 24,
        max_results: int = 10,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Add an arXiv source to check periodically.
        
        Args:
            source_id: Unique identifier for the source
            search_query: arXiv search query
            categories: List of arXiv categories to search in
            interval_hours: Interval in hours between checks
            max_results: Maximum number of results to return
            metadata: Optional metadata about the source
            
        Returns:
            Status message
        """
        if source_id in self.sources:
            return f"Source with ID '{source_id}' already exists"
            
        # Create metadata if not provided
        if metadata is None:
            metadata = {}
            
        # Set default categories if not provided
        if categories is None:
            categories = ["cs.AI", "cs.CL", "cs.CV", "cs.LG"]
            
        # Add source info to metadata
        metadata.update({
            "source_id": source_id,
            "source_type": "arxiv",
            "search_query": search_query,
            "categories": categories,
            "first_check": datetime.datetime.now().isoformat()
        })
        
        # Define the job function
        def check_arxiv():
            try:
                from urllib.parse import quote
                
                metadata["last_check"] = datetime.datetime.now().isoformat()
                self.logger.info(f"Checking arXiv source: {source_id} - {search_query}")
                
                # Build the arXiv API query
                category_query = " OR ".join([f"cat:{cat}" for cat in categories])
                
                # Format the arXiv API URL
                url = f"http://export.arxiv.org/api/query?search_query=({quote(search_query)}) AND ({category_query})&sortBy=submittedDate&sortOrder=descending&max_results={max_results}"
                
                # Use the URL retriever to fetch and process the XML content
                result = self.research_agent.learn_from_url(url, metadata=metadata)
                
                self.logger.info(f"Result for {source_id}: {result}")
                return result
            except Exception as e:
                self.logger.error(f"Error checking arXiv source {source_id}: {str(e)}")
                return f"Error checking arXiv source {source_id}: {str(e)}"
        
        # Add the job to the scheduler
        job = self.scheduler.add_job(
            check_arxiv,
            trigger=IntervalTrigger(hours=interval_hours),
            id=source_id,
            replace_existing=True,
            next_run_time=datetime.datetime.now()  # Run immediately first time
        )
        
        # Store the source
        self.sources[source_id] = {
            "id": source_id,
            "type": "arxiv",
            "search_query": search_query,
            "categories": categories,
            "interval_hours": interval_hours,
            "max_results": max_results,
            "metadata": metadata,
            "job_id": job.id,
            "added_at": datetime.datetime.now().isoformat()
        }
        
        return f"Successfully added arXiv source '{source_id}' with interval {interval_hours} hours"
        
    def add_custom_source(
        self,
        source_id: str,
        check_function: Callable,
        interval_minutes: int = 60,
        cron_expression: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Add a custom source with a user-defined check function.
        
        Args:
            source_id: Unique identifier for the source
            check_function: Function to call to check the source
            interval_minutes: Interval in minutes between checks (ignored if cron_expression is provided)
            cron_expression: Cron expression for scheduling (optional)
            metadata: Optional metadata about the source
            
        Returns:
            Status message
        """
        if source_id in self.sources:
            return f"Source with ID '{source_id}' already exists"
            
        # Create metadata if not provided
        if metadata is None:
            metadata = {}
            
        # Add source info to metadata
        metadata.update({
            "source_id": source_id,
            "source_type": "custom",
            "first_check": datetime.datetime.now().isoformat()
        })
        
        # Define the job function wrapper
        def check_custom():
            try:
                metadata["last_check"] = datetime.datetime.now().isoformat()
                self.logger.info(f"Running custom source check: {source_id}")
                
                # Call the user-defined function
                result = check_function(self.research_agent, metadata)
                
                self.logger.info(f"Result for {source_id}: {result}")
                return result
            except Exception as e:
                self.logger.error(f"Error checking custom source {source_id}: {str(e)}")
                return f"Error checking custom source {source_id}: {str(e)}"
        
        # Add the job to the scheduler
        if cron_expression:
            trigger = CronTrigger.from_crontab(cron_expression)
            schedule_info = f"cron: {cron_expression}"
        else:
            trigger = IntervalTrigger(minutes=interval_minutes)
            schedule_info = f"interval: {interval_minutes} minutes"
            
        job = self.scheduler.add_job(
            check_custom,
            trigger=trigger,
            id=source_id,
            replace_existing=True,
            next_run_time=datetime.datetime.now()  # Run immediately first time
        )
        
        # Store the source
        self.sources[source_id] = {
            "id": source_id,
            "type": "custom",
            "schedule_info": schedule_info,
            "metadata": metadata,
            "job_id": job.id,
            "added_at": datetime.datetime.now().isoformat()
        }
        
        return f"Successfully added custom source '{source_id}' with {schedule_info}"
        
    def remove_source(self, source_id: str) -> str:
        """
        Remove a source from the scheduler.
        
        Args:
            source_id: ID of the source to remove
            
        Returns:
            Status message
        """
        if source_id not in self.sources:
            return f"Source with ID '{source_id}' not found"
            
        try:
            # Remove the job from the scheduler
            self.scheduler.remove_job(source_id)
            
            # Remove the source from the sources dictionary
            del self.sources[source_id]
            
            return f"Successfully removed source '{source_id}'"
        except JobLookupError:
            # Remove the source from the sources dictionary
            del self.sources[source_id]
            return f"Source '{source_id}' was not scheduled but has been removed from the sources list"
        except Exception as e:
            return f"Error removing source '{source_id}': {str(e)}"
            
    def get_source(self, source_id: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a source.
        
        Args:
            source_id: ID of the source
            
        Returns:
            Source information or None if not found
        """
        return self.sources.get(source_id)
        
    def get_all_sources(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about all sources.
        
        Returns:
            Dictionary of source information
        """
        return self.sources
        
    def update_source_interval(self, source_id: str, interval_minutes: int) -> str:
        """
        Update the interval for a source.
        
        Args:
            source_id: ID of the source
            interval_minutes: New interval in minutes
            
        Returns:
            Status message
        """
        if source_id not in self.sources:
            return f"Source with ID '{source_id}' not found"
            
        try:
            # Update the job in the scheduler
            self.scheduler.reschedule_job(
                source_id,
                trigger=IntervalTrigger(minutes=interval_minutes)
            )
            
            # Update the source information
            if self.sources[source_id]["type"] == "arxiv":
                self.sources[source_id]["interval_hours"] = interval_minutes / 60
            else:
                self.sources[source_id]["interval_minutes"] = interval_minutes
            
            self.sources[source_id]["schedule_info"] = f"interval: {interval_minutes} minutes"
            
            return f"Successfully updated interval for source '{source_id}' to {interval_minutes} minutes"
        except Exception as e:
            return f"Error updating interval for source '{source_id}': {str(e)}"
            
    def add_prompt_on_url_task(
        self,
        task_id: str,
        url: str,
        prompt_template: str,
        output_action: Dict[str, Any],
        interval_minutes: int = 60,
        cron_expression: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Add a scheduled task to fetch content from a URL, execute a prompt, and act on the result.

        Args:
            task_id: Unique identifier for the task.
            url: URL to fetch content from.
            prompt_template: Template for the prompt. Use {url_content} as placeholder for fetched content.
            output_action: Configuration for what to do with the prompt's result.
                         Example: {"type": "log_to_file", "filepath": "./task_logs/log.txt"}
            interval_minutes: Interval in minutes between checks (if not using cron).
            cron_expression: Cron expression for scheduling (overrides interval_minutes).
            metadata: Optional metadata about the task.

        Returns:
            Status message.
        """
        if task_id in self.scheduled_tasks or task_id in self.sources:
            return f"Task or Source with ID '{task_id}' already exists"

        if metadata is None:
            metadata = {}
        metadata.update({
            "task_id": task_id,
            "task_type": "prompt_on_url",
            "url": url,
            "first_check": datetime.datetime.now().isoformat()
        })

        def execute_prompt_on_url():
            try:
                self.logger.info(f"Executing scheduled task: {task_id} - for URL: {url}")
                metadata["last_check"] = datetime.datetime.now().isoformat()

                # 1. Fetch URL content
                self.logger.debug(f"Task {task_id} - Fetching content from URL: {url}")
                url_content_str = "Error: Failed to fetch URL content."
                try:
                    loader = WebBaseLoader([url]) # WebBaseLoader expects a list
                    docs = loader.load()
                    if docs:
                        url_content_str = "\n".join([doc.page_content for doc in docs])
                        self.logger.info(f"Task {task_id} - Fetched content length: {len(url_content_str)}. Preview: {url_content_str[:200]}...")
                        if not url_content_str.strip():
                            self.logger.warning(f"Task {task_id} - Fetched URL content is empty or whitespace only.")
                    else:
                        self.logger.warning(f"Task {task_id} - WebBaseLoader returned no documents for URL: {url}")
                        url_content_str = "Error: No documents returned by WebBaseLoader."
                except Exception as fetch_exc:
                    self.logger.error(f"Task {task_id} - Exception during URL content fetching: {fetch_exc}", exc_info=True)
                    # url_content_str already defaults to an error message

                # 2. Format the prompt
                final_prompt = prompt_template.format(url=url, url_content=url_content_str)
                self.logger.debug(f"Task {task_id} - Constructed final_prompt (first 200 chars of content): {prompt_template.format(url=url, url_content=url_content_str[:200]+'...')[:400]}...")

                # 3. Execute the prompt using a direct LLM call
                output_result = "Error: LLM invocation did not produce a usable result."
                self.logger.info(f"Task {task_id} - Preparing for direct LLM invocation.")
                try:
                    self.logger.info(f"Task {task_id} - Attempting self.research_agent.llm.invoke(final_prompt)")
                    llm_response = self.research_agent.llm.invoke(final_prompt)
                    self.logger.info(f"Task {task_id} - LLM invoke call returned. Type of response: {type(llm_response)}")
                    
                    if hasattr(llm_response, 'content'):
                        output_result = llm_response.content
                        self.logger.info(f"Task {task_id} - Extracted content from LLM response: {output_result[:500]}...")
                    else:
                        output_result = str(llm_response)
                        self.logger.warning(f"Task {task_id} - LLM response object did not have 'content' attribute. Stringified: {output_result[:500]}...")
                    
                    if not output_result.strip():
                         self.logger.warning(f"Task {task_id} - LLM output_result is empty or whitespace only after extraction.")

                except Exception as llm_exc:
                    self.logger.error(f"Task {task_id} - Exception during direct LLM invocation or result processing: {llm_exc}", exc_info=True)
                    output_result = f"Error during LLM invocation/processing: {str(llm_exc)}"

                # 4. Perform output action
                action_type = output_action.get("type")
                self.logger.info(f"Task {task_id} - Preparing to perform output action: {action_type} with result preview: {output_result[:200]}...")

                if action_type == "log_to_file":
                    filepath = output_action.get("filepath", f"./tasks_out/{task_id}.out")
                    try:
                        os.makedirs(os.path.dirname(filepath), exist_ok=True)
                        log_entry_url_content = url_content_str[:500] + '...' if len(url_content_str) > 500 else url_content_str
                        log_entry = f"[{datetime.datetime.now().isoformat()}] Task: {task_id}\nURL: {url}\nPrompt: {prompt_template.format(url=url, url_content=log_entry_url_content)}\nResponse: {output_result}\n---\n"
                        self.logger.info(f"Task {task_id} - Attempting to write log_entry (length {len(log_entry)}) to file: {filepath}")
                        with open(filepath, "a") as f:
                            f.write(log_entry)
                        self.logger.info(f"Task {task_id} - Successfully wrote log_entry to {filepath}")
                    except Exception as file_write_exc:
                        self.logger.error(f"Task {task_id} - Exception during file write to {filepath}: {file_write_exc}", exc_info=True)
                
                elif action_type == "add_to_knowledge_base":
                    doc_title = output_action.get("document_title", f"Scheduled Task Output: {task_id} - {url}")
                    doc_title = doc_title.format(url=url, date=datetime.datetime.now().strftime("%Y-%m-%d"))
                    kb_metadata = output_action.get("metadata", {})
                    import hashlib
                    content_len = len(output_result)
                    content_h = hashlib.sha256(output_result.encode('utf-8')).hexdigest()

                    kb_metadata.update({
                        "document_id": url, # Use URL as the document ID
                        "original_task_id": task_id,
                        "original_url": url,
                        "generation_date": datetime.datetime.now().isoformat(),
                        "source_type": "scheduled_prompt_on_url",
                        "document_title": doc_title,
                        "content_length": content_len,
                        "content_hash": content_h
                    })
                    self.logger.info(f"Task {task_id} - Attempting to add to knowledge base. Title: {doc_title}, ID: {url}, Length: {content_len}, Hash: {content_h}")
                    if output_result and output_result.strip() and "Error: " not in output_result:
                        # Pass the document_id explicitly if add_text supports it, or ensure it's in metadata
                        # For now, assuming add_text will pick up 'document_id' from metadata for Chroma's ID system.
                        status_msg = self.research_agent.retriever.add_text(text=output_result, metadata=kb_metadata)
                        self.logger.info(f"Task {task_id} - Add to knowledge base status for '{doc_title}': {status_msg}")
                    else:
                        self.logger.warning(f"Task {task_id} - Skipping add_to_knowledge_base due to empty or error in output_result.")
                else:
                    self.logger.warning(f"Task {task_id} - Unknown or no output action type specified: {action_type}")
                
                # This return is for the execute_prompt_on_url function itself, within the main try block.
                # It's not an HTTP response or anything, just a signal for APScheduler if needed.
                # The actual result is handled by the output_action.
                self.logger.info(f"Task {task_id} completed processing.")
            except Exception as e:
                self.logger.error(f"Error executing scheduled task {task_id}: {str(e)}", exc_info=True)
                return f"Error executing task {task_id}: {str(e)}"

        if cron_expression:
            trigger = CronTrigger.from_crontab(cron_expression)
            schedule_info = f"cron: {cron_expression}"
        else:
            trigger = IntervalTrigger(minutes=interval_minutes)
            schedule_info = f"interval: {interval_minutes} minutes"

        job = self.scheduler.add_job(
            execute_prompt_on_url,
            trigger=trigger,
            id=task_id,
            replace_existing=True,
            next_run_time=datetime.datetime.now()  # Run immediately first time
        )

        self.scheduled_tasks[task_id] = {
            "id": task_id,
            "type": "prompt_on_url",
            "url": url,
            "prompt_template": prompt_template,
            "output_action": output_action,
            "schedule_info": schedule_info,
            "metadata": metadata,
            "job_id": job.id,
            "added_at": datetime.datetime.now().isoformat()
        }
        return f"Successfully added scheduled task '{task_id}' with {schedule_info}"

    def get_all_scheduled_tasks(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about all scheduled tasks.
        
        Returns:
            Dictionary of scheduled task information.
        """
        return self.scheduled_tasks
        
    def remove_scheduled_task(self, task_id: str) -> str:
        """
        Remove a scheduled task.
        
        Args:
            task_id: ID of the task to remove
            
        Returns:
            Message indicating success or failure
            
        Raises:
            ValueError: If the task doesn't exist
        """
        if task_id not in self.scheduled_tasks:
            raise ValueError(f"Task '{task_id}' not found in scheduled tasks")
            
        task = self.scheduled_tasks[task_id]
        job_id = task.get('job_id')
        
        # Remove the job from APScheduler
        if job_id and self.scheduler.get_job(job_id):
            self.scheduler.remove_job(job_id)
            self.logger.info(f"Removed job {job_id} from APScheduler")
        
        # Remove from our tasks dictionary
        del self.scheduled_tasks[task_id]
        self.logger.info(f"Removed task {task_id} from scheduled tasks dictionary")
        
        return f"Successfully removed scheduled task '{task_id}'"

    def shutdown(self):
        """Shut down the scheduler."""
        self.scheduler.shutdown()
