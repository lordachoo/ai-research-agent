# Project Brief: AI Research Agent

## Core Objective
To develop an AI-powered research assistant capable of autonomously gathering, processing, and synthesizing information from various sources, and maintaining an organized, queryable knowledge base.

## Key Requirements
1.  **Information Ingestion**:
    *   Fetch content from URLs.
    *   Process local documents (PDF, DOCX, TXT, MD, CSV).
    *   Integrate with sources like arXiv for academic papers.
2.  **Content Processing**:
    *   Utilize Large Language Models (LLMs) for tasks like summarization, question answering, and analysis based on ingested content.
    *   Support customizable prompts for directing LLM tasks.
3.  **Knowledge Base Management**:
    *   Store processed information in a persistent vector database (ChromaDB).
    *   Implement robust deduplication to avoid redundant entries and ensure data integrity (based on source URL and content hash).
    *   Allow intelligent updates to existing knowledge base entries when source content changes.
4.  **Scheduled Operations**:
    *   Enable users to schedule periodic checks of specified sources (URLs, arXiv queries).
    *   Allow scheduling of custom "prompt-on-URL" tasks where the LLM processes content from a URL and performs a defined output action (e.g., log to file, add to knowledge base).
5.  **User Interaction**:
    *   Provide a Command-Line Interface (CLI) for all core functionalities (querying, adding sources, scheduling, etc.).
    *   (Future/Optional) Offer a RESTful API for programmatic interaction.
6.  **Configuration & Control**:
    *   Manage LLM settings (provider, model, temperature) via environment variables.
    *   Use JSON configuration files for defining scheduled sources and tasks.
    *   Ensure LLM output determinism (e.g., temperature 0.0) for consistent hashing and deduplication where needed.

## Scope
-   **In Scope**: Core functionalities listed above, modular design, error handling, basic logging.
-   **Out of Scope (Initially)**: Advanced UI/web interface, complex multi-agent systems, real-time collaborative features.

## Success Criteria
-   The agent can successfully ingest, process, and store information from specified sources.
-   The knowledge base correctly handles duplicates and updates.
-   Scheduled tasks execute reliably and perform their defined actions.
-   Users can effectively interact with the agent via the CLI.
