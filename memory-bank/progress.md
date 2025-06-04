# Progress: AI Research Agent

## What Works (Key Implemented Features)
1.  **Core Agent & LLM Interaction**:
    *   `ResearchAgent` can successfully invoke LLMs (Ollama local models, OpenAI cloud models) for prompt execution.
    *   LLM `TEMPERATURE` is configurable via `.env` and correctly passed to the LLM, enabling deterministic output (critical for content hashing).
2.  **Information Ingestion**:
    *   Fetching and parsing content from URLs using LangChain's `WebBaseLoader`.
    *   Loading local documents (tested with various types, functionality provided by LangChain loaders).
3.  **Knowledge Base**:
    *   Persistent knowledge base using ChromaDB, storing document chunks and their embeddings.
    *   **Advanced Deduplication & Update Logic**:
        *   `KnowledgeBase.add_documents` successfully uses `document_id` (URL) and `content_hash` (SHA256 of LLM summary) for managing entries.
        *   Skips adding summaries if the exact URL and content hash already exist.
        *   Deletes old chunks and adds new ones if the content hash for a known URL changes (update).
    *   Metadata (including `document_id`, `content_hash`, `content_length`, `source_type`, `title`) is stored with document chunks.
4.  **Scheduled Tasks (`SourceScheduler`)**:
    *   Successfully schedules and executes "prompt-on-URL" tasks based on JSON configuration.
    *   Supports `interval_minutes` and `cron_expression` for scheduling.
    *   Calculates `content_hash` and `content_length` for LLM outputs from scheduled tasks.
    *   Output actions:
        *   `log_to_file`: Works as expected.
        *   `add_to_knowledge_base`: Integrates with the new deduplication/update logic in `KnowledgeBase`.
5.  **CLI (`app.main.py`)**:
    *   Provides commands for querying, adding documents/URLs, and running the scheduler.
    *   Loads configurations from `.env` and JSON files.
6.  **Configuration**:
    *   `.env` file for API keys, LLM settings.
    *   JSON files for defining scheduled sources and tasks.
    *   `.env.example` provides a clear template.
7.  **README & Documentation**:
    *   `README.md` updated with recent features, usage instructions for scheduled tasks, and deduplication details.
    *   This Memory Bank (`/memory-bank` directory with Markdown files) is being established.

## What's Left to Build / Next Major Steps
1.  **Thorough Deduplication Testing (Ongoing)**:
    *   Systematically test the add/skip/update scenarios for `add_to_knowledge_base` to confirm flawless operation.
    *   Investigate any lingering anomalies (like the "already exists in DB" message on a supposedly fresh KB).
2.  **arXiv Integration**: While `SourceScheduler` has placeholders, the actual arXiv fetching and processing logic needs full implementation and testing if it's a priority.
3.  **General Source Monitoring**: Implement the logic for periodically checking generic "sources" (defined in JSON config) for new content to add to the main KB (distinct from specific prompt-on-URL task outputs).
4.  **Query Refinement**: Enhance the querying mechanism. This might involve:
    *   More sophisticated retrieval strategies (e.g., MMR, similarity score thresholds).
    *   Allowing users to specify search kwargs.
    *   Better formatting of query results.
5.  **API Development (`app.api.py`)**: Flesh out the FastAPI application to provide robust API endpoints for all key agent functionalities.
6.  **Error Handling & Resilience**: Continue to improve error handling across all components, especially for network operations and external API calls.
7.  **Unit & Integration Tests**: Develop a suite of tests to ensure reliability and prevent regressions.

## Current Status
-   The core functionality for scheduled "prompt-on-URL" tasks with intelligent knowledge base integration (deduplication and updates) is largely complete and in the process of final testing.
-   The foundational "Memory Bank" documentation structure is being created.
-   The project is stable enough for focused testing of the knowledge base features.

## Known Issues & Observations
-   **Previous KB Test Anomaly**: During a recent test, logs indicated "Summary for URL ... already exists in DB" even after attempting to clear the `./knowledge_base` directory. This needs to be re-verified with a guaranteed clean KB directory. It's possible the directory wasn't fully deleted or ChromaDB has some subtle persistence behavior if not shut down cleanly (though unlikely with file-based persistence).
-   **Logging Counts for KB**: The `added_count` and `updated_count` in `KnowledgeBase.add_documents` log messages currently reflect chunks processed for new/updated URLs, not strictly unique URLs. This is a minor logging detail; the database operations are the critical part.

## Evolution of Project Decisions
-   **Initial KB `add`**: Simple additions, risking duplicates.
-   **ID-based `get`**: Attempted deduplication using Chroma's `get(ids=[doc_id])`, but this was insufficient for chunked documents and didn't handle content updates well.
-   **Metadata Filtering (`where` clause)**: Shifted to using `document_id` (URL) and `content_hash` in metadata, queried via `vector_store.get(where={...})`. This proved much more effective for identifying true duplicates and changes.
-   **Explicit Deletion for Updates**: Realized that for an "update" (URL exists, content hash changes), old chunks for that URL must be explicitly deleted before adding new ones to prevent stale data.
-   **LLM Temperature to 0.0**: Recognized the necessity of deterministic LLM output for stable content hashes, leading to the enforcement of `TEMPERATURE=0.0` for relevant tasks.
