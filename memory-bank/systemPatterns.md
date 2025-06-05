# System Patterns: AI Research Agent

## System Architecture Overview
The AI Research Agent is a Python-based application with a modular architecture. It's designed to be run primarily via a Command-Line Interface (CLI), with distinct components for different aspects of its functionality.

-   **Core Logic (`app.core`)**: Contains the central classes for agent behavior, knowledge base interaction, and document retrieval.
    -   `agent.py` (`ResearchAgent`): Orchestrates LLM interactions, prompt formatting, and potentially agentic ReAct loops (depending on LLM and configuration).
    -   `knowledge_base.py` (`KnowledgeBase`): Manages all interactions with the ChromaDB vector store. This includes adding new documents (with deduplication and update logic), retrieving context for queries, and managing document chunks.
    -   `retriever.py` (`DocumentRetriever`): Handles fetching content from various sources (URLs, local files), parsing them, splitting them into manageable chunks (using LangChain's text splitters), and preparing them with metadata for the `KnowledgeBase`.
-   **Scheduling (`app.schedulers`)**:
    -   `source_scheduler.py` (`SourceScheduler`): Uses APScheduler to manage and execute scheduled tasks defined in JSON configuration files. This includes both general source monitoring and specific "prompt-on-URL" tasks. It calculates metadata like `content_hash` and `content_length` for LLM outputs.
-   **Main Application (`app.main.py`)**:
    -   Uses Typer to provide the CLI commands (`query`, `add-document`, `add-url`, `schedule`, `api`).
    -   Initializes the `ResearchAgent`, `KnowledgeBase`, and other core components.
    -   Loads environment variables (e.g., for API keys, LLM settings) using `python-dotenv`.
    -   Parses JSON configuration files for scheduled tasks.
-   **API (`app.api.py`)**: (Currently a placeholder or basic implementation) Intended to expose agent functionalities via a RESTful API using FastAPI.

## Key Technical Decisions & Design Patterns
-   **Modular Design**: Each major function (LLM interaction, KB management, scheduling, CLI) is handled by a dedicated module/class, promoting separation of concerns.
-   **Configuration-Driven**: System behavior, especially for scheduling and LLM settings, is largely controlled by external configuration files (`.env`, JSON configs) rather than hardcoding.
-   **LangChain Integration**: Heavily leverages the LangChain library for:
    -   LLM wrappers (`ChatOllama`, `ChatOpenAI`).
    -   Document loaders (for various file types and URLs).
    -   Text splitters (e.g., `RecursiveCharacterTextSplitter`).
    -   Vector store integration (ChromaDB via `Chroma`).
    -   Prompt templates.
-   **Metadata-Driven Deduplication**: The knowledge base uses `document_id` (source URL) and `content_hash` (SHA256 of LLM-generated summary) in document metadata to manage uniqueness and updates, rather than relying solely on vector similarity or Chroma's internal IDs for this application-level concern.
-   **Explicit LLM Temperature Control**: The system ensures the `TEMPERATURE` setting (especially `0.0` for determinism) is passed from configuration down to the LLM instantiation to control output variability.
-   **CLI First**: Primary interaction is through a Typer-based CLI, making it accessible for scripting and direct use.

## Component Relationships
-   `app.main.py` acts as the entry point, initializing and invoking other components.
-   `SourceScheduler` uses `ResearchAgent` to execute prompts and `DocumentRetriever` (which in turn uses `KnowledgeBase`) to add outputs to the KB.
-   `ResearchAgent` can use `KnowledgeBase` (via a retriever) to fetch context for answering queries.
-   `DocumentRetriever` uses `KnowledgeBase.add_documents` to store processed text.

## Critical Implementation Paths
-   **Scheduled Task Execution (`prompt_on_url` -> `add_to_knowledge_base`)**:
    1.  `SourceScheduler` triggers a task.
    2.  Fetches URL content.
    3.  `ResearchAgent.llm.invoke()` processes content with the user's prompt.
    4.  `SourceScheduler` calculates `content_hash` of the LLM output.
    5.  `DocumentRetriever.add_text()` is called with the LLM output and metadata (including `document_id`=URL, `content_hash`).
    6.  `DocumentRetriever` creates LangChain `Document` objects, splits them.
    7.  `KnowledgeBase.add_documents()` is called.
    8.  `KnowledgeBase` performs deduplication/update logic using metadata filters against ChromaDB, then adds/skips/updates chunks.
-   **Querying the Knowledge Base**:
    1.  User issues a query via CLI, Web UI or API.
    2.  `app.main.py` calls `ResearchAgent.run()` with optional `use_knowledge_base` parameter.
    3.  If knowledge base is enabled:
        - The agent directly queries the knowledge base using the KB tool
        - Retrieved KB results are incorporated into a prompt sent to the LLM
    4.  If knowledge base is disabled:
        - The agent bypasses KB retrieval and sends the query directly to the LLM
        - No KB context is included in the prompt
    5.  LLM generates an answer based on available context (with or without KB).
