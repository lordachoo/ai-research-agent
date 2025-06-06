# Active Context: AI Research Agent

## Current Work Focus
The primary focus is on enhancing the knowledge base UI, improving URL recursion, and implementing better content extraction:

1. **Knowledge Base UI Improvements**:
   - Added comprehensive metadata display for each chunk in a collapsible section
   - Implemented a table format showing all metadata fields (title, language, document_id, etc.)

2. **URL Recursion Enhancements**:
   - Removed domain restriction during URL crawling to follow all links (previously only same-domain)
   - Fixed logging messages related to link extraction
   - Removed the arbitrary 10-link limit per page
   - Increased recursion depth limit from 3 to 10 in the UI form

3. **Enhanced Content Extraction**:
   - Implemented improved content extraction using advanced BeautifulSoup parsing
   - Added UI checkbox option for enabling enhanced content extraction
   - Focused on extracting main content while filtering navigation, ads, etc.

Secondary focus continues to be on the knowledge base deduplication and update logic:
-   LLM-generated summaries added via `add_to_knowledge_base` are correctly identified by their source URL (`document_id`) and content hash (`content_hash`).
-   Duplicate summaries (same URL, same content hash) are skipped.
-   Updated summaries (same URL, different content hash) result in the deletion of old chunks for that URL and the addition of new chunks.
-   The LLM `TEMPERATURE` setting of `0.0` is consistently applied to ensure deterministic summary generation, which is crucial for reliable content hashing.

A secondary, immediate task is the creation of this structured Memory Bank (physical Markdown files in the `/memory-bank` directory) as per the `memorybank.md` specification.

## Recent Changes & Key Decisions
-   **Knowledge Base Deduplication Refactor**: `app.core.knowledge_base.KnowledgeBase.add_documents` was significantly rewritten to use metadata filtering (`document_id`, `content_hash`) in ChromaDB queries.
    -   Uses `vector_store.get(where={"$and": [{"document_id": "url"}, {"content_hash": "hash"}]})` to check for exact duplicates.
    -   Uses `vector_store.get(where={"document_id": "url"})` to find all versions for a URL, then `vector_store.delete(ids=[...])` to remove old chunks before adding new ones if the content hash has changed.
-   **LLM Determinism**: Confirmed and enforced `TEMPERATURE=0.0` in `.env` and `.env.example`, passed through `app.main.py` -> `ResearchAgent` -> `ChatOllama`, for consistent LLM output.
-   **Metadata Enrichment**: `app.schedulers.source_scheduler.py` now calculates and adds `content_length` and `content_hash` (SHA256 of LLM output) to document metadata before adding to the knowledge base. `document_id` is set to the source URL.
-   **README Update**: The main `README.md` was updated to reflect these new deduplication features and the importance of the `TEMPERATURE` setting.
-   **Memory Bank Creation**: Initiated the process of creating physical Markdown files for the Memory Bank as per `memorybank.md`. Explicitly created the `memory-bank` directory after `write_to_file` initially failed.

## Next Steps (Immediate)
1.  Complete the creation of all core Memory Bank Markdown files (`projectbrief.md`, `productContext.md`, `activeContext.md`, `systemPatterns.md`, `techContext.md`, `progress.md`).
2.  Conduct thorough testing of the knowledge base deduplication:
    *   Delete the existing `./knowledge_base` directory for a clean test.
    *   Run a scheduled task (e.g., `example_schedule_config-addToKnowledgeBase.json`) multiple times to observe:
        *   First run: New additions.
        *   Second run (identical content): Skipped duplicates.
        *   Third run (modified prompt to change summary content): Deletion of old, addition of new (update).
    *   Carefully examine logs for expected behavior.
3.  Address any anomalies observed during testing (e.g., the previous observation of "already exists in DB" on what was thought to be a fresh KB).

## Active Decisions & Considerations
-   The current deduplication logic counts "added" and "updated" based on chunks processed rather than unique URLs per batch. This is a simplification for logging but the underlying DB operations (skip, delete-then-add) are what matter most.
-   Relying on ChromaDB's internal ID generation for chunks, while using our metadata (`document_id`, `content_hash`) for application-level deduplication and update logic.

## Important Patterns & Preferences
-   **Modularity**: Keep components like `KnowledgeBase`, `ResearchAgent`, and `SourceScheduler` distinct and focused.
-   **Configuration over Code**: Use `.env` files and JSON configurations for settings and task definitions.
-   **Clear Logging**: Provide informative log messages to track agent behavior and data processing.
-   **Determinism for KB**: Emphasize `TEMPERATURE=0.0` for tasks that feed into the knowledge base where content stability is key for deduplication.
