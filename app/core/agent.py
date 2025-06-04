"""
Core agent implementation for the AI Research Agent.
This module handles the creation and management of the AI agent.
"""

import os
from typing import List, Dict, Any, Optional
from langchain.agents import AgentExecutor, create_openai_tools_agent, create_react_agent
from langchain import hub
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain.tools import BaseTool

from app.core.retriever import DocumentRetriever
from app.core.knowledge_base import KnowledgeBase


class ResearchAgent:
    """AI Research Agent that can be trained on documents and perform research tasks."""

    def __init__(
        self,
        agent_name: str,
        knowledge_base: KnowledgeBase,
        model_name: str = "gpt-3.5-turbo-0125", # Default OpenAI model
        temperature: float = 0.1,
        llm_provider: str = "openai",
        ollama_base_url: str = "http://localhost:11434",
        ollama_model_name: str = "llama3:latest", # Default Ollama chat model
    ):
        """
        Initialize the Research Agent.

        Args:
            agent_name: Name of the agent
            knowledge_base: KnowledgeBase instance for document storage and retrieval
            model_name: OpenAI model name to use
            temperature: Temperature parameter for the model
        """
        self.agent_name = agent_name
        self.knowledge_base = knowledge_base
        self.temperature = temperature
        self.llm_provider = llm_provider.lower()

        if self.llm_provider == "ollama":
            self.model_name = ollama_model_name
            self.llm = ChatOllama(
                model=self.model_name,
                base_url=ollama_base_url,
                temperature=self.temperature
            )
        else: # Default to OpenAI
            self.model_name = model_name
            self.llm = ChatOpenAI(model=self.model_name, temperature=self.temperature)
        self.tools = []
        self.agent_executor = None
        self.retriever = DocumentRetriever(knowledge_base)
        
        # Initialize default tools
        self._initialize_tools()
        
    def _initialize_tools(self):
        """Initialize the default set of tools for the agent."""
        # Add the retriever tool for accessing the knowledge base
        retriever_tool = self.retriever.get_retriever_tool()
        self.tools.append(retriever_tool)
        
        # Initialize the agent with the tools
        self._create_agent()
        
    def _create_agent(self):
        """Create the agent executor with the defined tools, choosing agent type based on LLM provider."""
        if self.llm_provider == "ollama":
            # For Ollama, use a ReAct agent which is more broadly compatible
            # Pull a ReAct prompt template from Langchain Hub
            # This prompt is designed for chat models and ReAct logic.
            # It expects 'input', 'chat_history', and 'agent_scratchpad' as input variables.
            # It also includes placeholders for 'tools' and 'tool_names' in its system message part.
            prompt = hub.pull("hwchase17/react-chat")
            agent = create_react_agent(self.llm, self.tools, prompt)
        else:
            # For OpenAI, use the tools agent as before
            system_prompt = f"""You are {self.agent_name}, an AI research assistant specialized in providing accurate information.
        
        You have access to a knowledge base of documents that you've been trained on.
        When answering questions, always try to use the information from your knowledge base first.
        
        If you don't know the answer or can't find relevant information in your knowledge base, 
        clearly state that you don't have that information.
        
        Always cite your sources when providing information from your knowledge base.
        """
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ])
            agent = create_openai_tools_agent(self.llm, self.tools, prompt)
    
        # Create the agent executor
        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            return_intermediate_steps=True,
            # Add handling for parsing errors, which can be common with non-OpenAI models
            # This provides feedback to the LLM to correct its output format.
            handle_parsing_errors="Check your output and make sure it conforms to the expected format. For example, ensure all JSON strings are properly quoted."
        )

    def add_tool(self, tool: BaseTool):
        """
        Add a new tool to the agent.
        
        Args:
            tool: The tool to add
        """
        self.tools.append(tool)
        self._create_agent()  # Recreate the agent with the updated tools
        
    def run(self, query: str, chat_history: List = None) -> Dict[str, Any]:
        """
        Run the agent on a query.
        
        Args:
            query: The user query
            chat_history: Optional chat history for context
            
        Returns:
            The agent's response
        """
        if chat_history is None:
            chat_history = []
            
        response = self.agent_executor.invoke({
            "input": query,
            "chat_history": chat_history
        })
        
        return response
        
    def learn_from_document(self, document_path: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Learn from a document by adding it to the knowledge base.
        
        Args:
            document_path: Path to the document
            metadata: Optional metadata about the document
            
        Returns:
            Status message
        """
        return self.retriever.add_document(document_path, metadata)
    
    def learn_from_url(self, url: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Learn from content at a URL by adding it to the knowledge base.
        
        Args:
            url: URL to fetch and learn from
            metadata: Optional metadata about the content
            
        Returns:
            Status message
        """
        return self.retriever.add_url_content(url, metadata)
