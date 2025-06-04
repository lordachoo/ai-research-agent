# Product Context: AI Research Agent

## Problem Solved
The AI Research Agent addresses the challenge of information overload and the time-consuming nature of manual research. It automates the process of:
-   Discovering relevant information from diverse online and local sources.
-   Synthesizing and summarizing large volumes of text.
-   Maintaining an organized and up-to-date knowledge repository.
-   Periodically checking sources for new or updated information.

This allows users (researchers, analysts, students, lifelong learners) to stay informed more efficiently and focus on higher-level analysis rather than manual data collection and initial processing.

## How It Should Work (User Perspective)
1.  **Setup**: The user configures the agent with their preferred LLM (local via Ollama or cloud-based like OpenAI) and API keys through an `.env` file.
2.  **Adding Knowledge**:
    *   Users can point the agent to specific URLs or local documents (PDFs, text files, etc.) to be processed and added to its knowledge base.
    *   They can define "sources" (e.g., an arXiv query for "machine learning" or a specific blog URL) for the agent to monitor.
3.  **Scheduled Tasks**:
    *   Users can create JSON configuration files to define:
        *   **Source Monitoring**: How often to check defined sources for new content to add to the general knowledge base.
        *   **Prompt-on-URL Tasks**: Specific URLs to fetch periodically, a custom prompt to run against that content, and an action for the LLM's output (e.g., log to a file, or add the processed output as a new, distinct entry to the knowledge base).
4.  **Querying**: Users can ask the agent questions or request summaries based on the information it has accumulated in its knowledge base. The agent retrieves relevant context and uses its LLM to generate a response.
5.  **Interaction**: Primarily via a Command-Line Interface (CLI). Users run commands like `python -m app.main query "..."`, `python -m app.main add-url ...`, or `python -m app.main schedule config.json`.

## User Experience Goals
-   **Ease of Use**: Simple CLI commands and straightforward JSON configuration.
-   **Reliability**: Consistent performance of scheduled tasks and accurate information retrieval.
-   **Transparency**: Clear logging of agent actions and processing steps.
-   **Control**: Users should have control over data sources, LLM settings (especially temperature for deterministic outputs), and scheduling frequency.
-   **Efficiency**: Save users significant time in their research and information gathering workflows.
-   **Focused Knowledge**: The `add_to_knowledge_base` action for scheduled tasks should allow for building specific, curated datasets from LLM outputs, distinct from general source ingestion, with robust deduplication.
