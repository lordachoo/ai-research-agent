"""
Core agent implementation for the AI Research Agent.
This module handles the creation and management of the AI agent.
"""

import os
import logging
from typing import List, Dict, Any, Optional

from langchain.agents import AgentExecutor, create_openai_tools_agent, create_react_agent
from langchain import hub
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain.tools import BaseTool

from app.core.retriever import DocumentRetriever
from app.core.knowledge_base import KnowledgeBase

# Set up logging
logger = logging.getLogger(__name__)

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
        
    def _create_agent(self, force_kb_use=False):
        """Create the agent executor with the defined tools, choosing agent type based on LLM provider."""
        # Base template components that are common for both providers
        template = """
        You are an AI research assistant specialized in providing accurate information.
        
        When answering questions, follow these instructions carefully:
        {kb_instructions}
        
        You have access to the following tools:
        
        {tools}
        
        To use a tool, please use the following format:
        
        ```
        Thought: Do I need to use a tool? Yes
        Action: the action to take, should be one of [{tool_names}]
        Action Input: the input to the action
        Observation: the result of the action
        ```
        
        When you have a response for the human, or if you do not need to use a tool, you MUST use the format:
        
        ```
        Thought: Do I need to use a tool? No
        Final Answer: [your response here]
        ```
        """
        
        # Different KB instructions based on whether KB usage is forced
        if force_kb_use:
            logger.info("Creating agent with forced knowledge base usage")
            kb_instructions = """
            IMPORTANT: You MUST ALWAYS use the knowledge_base_search tool first for EVERY query.
            NEVER decide to skip using the knowledge base tool - your answers should ALWAYS be derived 
            first from knowledge base results, then supplemented with your own knowledge.
            If the knowledge base doesn't have relevant information, explicitly state this in your response.
            Always cite your sources when providing information from your knowledge base.
            """
        else:
            logger.info("Creating agent without forced knowledge base usage")
            kb_instructions = """
            When answering questions, use your general knowledge to provide the best possible answer.
            If you don't know the answer or can't find relevant information, 
            clearly state that you don't have that information.
            """
        
        # Create the agent based on the LLM provider
        from langchain.agents import AgentExecutor
        from langchain.agents.agent_types import AgentType
        from langchain.tools.base import BaseTool
        from langchain_core.prompts import PromptTemplate
        from langchain.schema import AgentAction, AgentFinish
        from typing import Union, List, Tuple, Any, Dict, Optional
        import re
        
        # For Ollama models, use a custom agent implementation that's more tolerant of parsing errors
        if self.llm_provider == "ollama":
            # Import necessary components
            from langchain.agents import AgentOutputParser, create_react_agent
            from langchain_core.prompts import PromptTemplate
            from langchain_core.messages import AIMessage, HumanMessage
            import re
            
            # Define a custom output parser that's very tolerant of formatting errors
            class ForgivingOutputParser(AgentOutputParser):
                def parse(self, text):
                    from langchain.agents.agent import AgentAction, AgentFinish
                    
                    # Try to find a Final Answer
                    if "Final Answer:" in text:
                        answer = text.split("Final Answer:")[-1].strip()
                        return AgentFinish(return_values={"output": answer}, log=text)
                    
                    # Look for Action and Action Input even with irregular formatting
                    action_match = re.search(r"\bAction\s*:?\s*([\w\s_]+)", text, re.IGNORECASE)
                    input_match = re.search(r"\bAction\s+Input\s*:?\s*(.+?)(?:\n|$)", text, re.IGNORECASE | re.DOTALL)
                    
                    # Try to extract tool name and tool input
                    if action_match:
                        action = action_match.group(1).strip()
                        action_input = input_match.group(1).strip() if input_match else ""
                        
                        # Clean up action input - if it's a dict-like string, parse it
                        if action_input.startswith('{') and action_input.endswith('}'): 
                            try:
                                # Try to evaluate as Python dict
                                import ast
                                action_input = ast.literal_eval(action_input)
                            except:
                                # If that fails, keep as string
                                pass
                        
                        return AgentAction(tool=action, tool_input=action_input, log=text)
                    
                    # If we can't find explicit Action/Action Input format but see a tool name
                    # followed by some input, try to extract that
                    for tool in self.tools:
                        tool_pattern = rf"\b{re.escape(tool.name)}\b[^\n]*?([\w\s\-]+)\??"
                        tool_match = re.search(tool_pattern, text, re.IGNORECASE)
                        if tool_match:
                            return AgentAction(
                                tool=tool.name, 
                                tool_input=tool_match.group(1).strip(),
                                log=text
                            )
                    
                    # Fallback: Just try to find any tool name mentioned and use query as input
                    for tool in self.tools:
                        if tool.name.lower() in text.lower():
                            return AgentAction(
                                tool=tool.name,
                                tool_input="Please provide information about this query",
                                log=text
                            )
                    
                    # If all else fails, treat it as a final answer
                    return AgentFinish(return_values={"output": text}, log=text)
                
                @property
                def tools(self):
                    return self._tools
                
                @tools.setter
                def tools(self, tools):
                    self._tools = tools
            
            # Create a simpler prompt template with explicit instructions
            ollama_template = """
            You are an AI research assistant. 
            {kb_instructions_placeholder}
            
            You have access to the following tools:
            
            {tools}
            
            STRICT FORMAT REQUIREMENTS:
            Use the following format EXACTLY:
            
            Question: the input question you must answer
            Thought: you should always think about what to do
            Action: [one of {tool_names}]
            Action Input: [input for the tool]
            Observation: [tool result]
            ... (repeat Thought/Action/Action Input/Observation as needed)
            Thought: I now know the final answer
            Final Answer: [your final answer to the question]
            
            Begin! Remember to ALWAYS follow the format above.
            
            Question: {input}
            {agent_scratchpad}
            """
            
            # Create a prompt that's optimized for Ollama - insert KB instructions directly
            formatted_template = ollama_template.replace("{kb_instructions_placeholder}", kb_instructions)
            
            # Create a prompt without the kb_instructions variable
            prompt = PromptTemplate(
                template=formatted_template,
                input_variables=["input", "agent_scratchpad", "tools", "tool_names"]
            )
            
            # Create a custom parser instance
            output_parser = ForgivingOutputParser()
            output_parser.tools = self.tools
            
            # Create the agent with our forgiving parser
            agent = create_react_agent(self.llm, self.tools, prompt, output_parser=output_parser)
            
            # Create the agent executor with very generous parsing error handling
            self.agent_executor = AgentExecutor(
                agent=agent,
                tools=self.tools,
                verbose=True,
                handle_parsing_errors="I'll try to fix the format. Please answer with the exact format I showed earlier.",
                max_iterations=5,
                return_intermediate_steps=True
            )
        else:  # For OpenAI models
            from langchain.agents import initialize_agent
            self.agent_executor = initialize_agent(
                self.tools,
                self.llm,
                agent=AgentType.OPENAI_FUNCTIONS,
                verbose=True,
                handle_parsing_errors=True
            )
            
        # Update the system message in the agent's prompt with our KB instructions
        if hasattr(self.agent_executor, 'agent') and hasattr(self.agent_executor.agent, 'llm_chain') and hasattr(self.agent_executor.agent.llm_chain, 'prompt'):
            if hasattr(self.agent_executor.agent.llm_chain.prompt, 'messages'):
                for i, message in enumerate(self.agent_executor.agent.llm_chain.prompt.messages):
                    if hasattr(message, 'prompt') and 'system' in str(message).lower():
                        # Get the original system message
                        orig_system = message.prompt.template
                        # Prepend our KB instructions
                        message.prompt.template = f"{kb_instructions}\n\n{orig_system}"
                        logger.info("Modified agent system message with KB instructions")
                        break
        
        logger.info(f"Created agent with LLM provider: {self.llm_provider}, Force KB use: {force_kb_use}")
        
    def add_tool(self, tool: BaseTool):
        """
        Add a new tool to the agent.
        
        Args:
            tool: The tool to add
        """
        self.tools.append(tool)
        self._create_agent()  # Recreate the agent with the updated tools
        
    def run(self, query, chat_history=None, use_knowledge_base=True):
        """
        Run the agent on a query.
        
        Args:
            query: The user query
            chat_history: Optional chat history for context
            use_knowledge_base: Whether to use the knowledge base for this query
            
        Returns:
            The agent's response
        """
        if chat_history is None:
            chat_history = []
            
        # Import required message types for direct LLM interaction
        from langchain_core.messages import HumanMessage
        
        # Store the original tools configuration
        original_tools = self.tools.copy()
        
        try:
            # Handle knowledge base toggle
            if use_knowledge_base:
                logger.info(f"Running query with knowledge base enabled: {query[:50]}...")
                
                # Find the knowledge base tool if it exists
                kb_tool = next((tool for tool in self.tools if tool.name == "knowledge_base_search"), None)
                
                if kb_tool:
                    # First, directly query the knowledge base
                    try:
                        logger.info("Directly querying knowledge base")
                        kb_results = kb_tool.invoke({"query": query})
                        logger.info(f"KB search results: {kb_results[:100]}...")
                        
                        # If we got results, use a different prompt approach with direct templates
                        # rather than using the agent framework which has parsing issues
                        prompt = f"""
                        You are an AI research assistant. Answer the following question based on the provided knowledge 
                        base search results. If the knowledge base doesn't have relevant information, 
                        state this clearly, then answer based on your general knowledge.
                        
                        Question: {query}
                        
                        Knowledge base search results:
                        {kb_results}
                        
                        Please provide a complete, accurate answer to the question.
                        """
                        
                        # Bypass the agent framework and directly query the LLM
                        logger.info("Querying LLM directly with KB-enhanced prompt")
                        result = self.llm.invoke([HumanMessage(content=prompt)])
                        return result.content
                    
                    except Exception as kb_error:
                        logger.warning(f"Error using knowledge base directly: {str(kb_error)}")
                        # Fall back to general LLM query
                        prompt = f"""
                        You are an AI research assistant. Answer the following question based on your general knowledge.
                        
                        Question: {query}
                        
                        Note: I tried to search a knowledge base for this question but encountered an error.
                        Please provide the best answer you can.
                        """
                        result = self.llm.invoke([HumanMessage(content=prompt)])
                        return result.content
                else:
                    logger.warning("Knowledge base tool not found, using general query")
                    prompt = f"""
                    You are an AI research assistant. Answer the following question based on your general knowledge.
                    
                    Question: {query}
                    
                    Note: Knowledge base search was requested but no knowledge base tool is available.
                    """
                    result = self.llm.invoke([HumanMessage(content=prompt)])
                    return result.content
            else:
                # If not using KB, just query the LLM directly
                logger.info(f"Running query without knowledge base: {query[:50]}...")
                prompt = f"""
                You are an AI research assistant. Answer the following question based on your general knowledge.
                
                Question: {query}
                """
                result = self.llm.invoke([HumanMessage(content=prompt)])
                return result.content
            
        except Exception as e:
            logger.exception(f"Error during agent execution: {str(e)}")
            return f"I encountered an error while processing your query: {str(e)}"
            
        finally:
            # Always restore the original tools configuration after execution
            if original_tools != self.tools:
                logger.info("Restoring original tools configuration")
                self.tools = original_tools
        
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
