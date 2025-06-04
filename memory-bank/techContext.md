# Tech Context: AI Research Agent

## Core Technologies
-   **Programming Language**: Python (version 3.10+ target, currently running on 3.13 based on venv path)
-   **Package Management**: `pip` with `requirements.txt`

## Key Libraries & Frameworks
-   **LangChain**: Core framework for LLM application development. Used for:
    -   LLM integration: `ChatOllama`, `ChatOpenAI`
    -   Document loading: `WebBaseLoader`, `PyPDFLoader`, `TextLoader`, `Docx2txtLoader`, `CSVLoader`
    -   Text splitting: `RecursiveCharacterTextSplitter`
    -   Vector store interaction: `Chroma` (as vector store wrapper)
    -   Prompt management: `PromptTemplate`, `ChatPromptTemplate`
    -   Chains (potentially, e.g., `load_qa_chain` or custom chains)
-   **ChromaDB (`chromadb`)**: Vector database used for persistent storage of document embeddings and metadata.
    -   Accessed via LangChain's `Chroma` wrapper.
    -   Configured for local persistence (e.g., in `./knowledge_base` directory).
    -   Uses metadata filtering (e.g., `where={"document_id": "...", "content_hash": "..."}`) for deduplication and update logic.
-   **Ollama (`ollama`)**: For running local LLMs (e.g., Llama3, Qwen2.5).
    -   Interacted with via `ChatOllama` from LangChain.
    -   Requires Ollama server to be running separately.
    -   Base URL configured in `.env` (`OLLAMA_BASE_URL`).
-   **OpenAI API**: For using OpenAI models (e.g., GPT-3.5-turbo, GPT-4).
    -   Interacted with via `ChatOpenAI` from LangChain.
    -   Requires `OPENAI_API_KEY` in `.env`.
-   **Typer**: For building the Command-Line Interface (CLI) in `app.main.py`.
-   **APScheduler**: For scheduling background tasks (source checking, prompt-on-URL execution) in `app.schedulers.source_scheduler.py`.
-   **Pydantic**: Used by LangChain and FastAPI (if API is developed) for data validation and settings management.
-   **Requests**: For direct HTTP calls if needed (though LangChain loaders often abstract this).
-   **Beautiful Soup (`beautifulsoup4`)**: Used by `WebBaseLoader` (and potentially directly) for parsing HTML content.
-   **python-dotenv**: For loading environment variables from `.env` files.
-   **FastAPI**: (Optional/Future) For building the RESTful API defined in `app.api.py`.

## Development & Runtime Environment
-   **Operating System**: Developed and tested on Linux.
-   **Python Virtual Environment**: Project uses a `venv` (e.g., `/home/anelson/ai-research-agent/venv/`) to manage dependencies.
-   **`.env` File**: Critical for configuration. Stores API keys, LLM provider choices, model names, Ollama base URL, LLM temperature, and LangSmith tracing settings. `.env.example` provides a template.
-   **JSON Configuration Files**: Used to define sources and scheduled tasks for the `SourceScheduler` (e.g., `example_schedule_config-addToKnowledgeBase.json`).
-   **Git**: For version control. `.gitignore` is used to exclude environment files, local data directories (`knowledge_base/`, `tasks_out/`), and Python caches.

## Technical Constraints & Considerations
-   **LLM Availability**: Functionality depends on access to either a running Ollama instance with downloaded models or valid OpenAI API keys.
-   **Rate Limiting**: When using external APIs (OpenAI, web scraping), be mindful of potential rate limits.
-   **Disk Space**: The `knowledge_base/` directory can grow depending on the amount of data processed.
-   **Processing Time**: Embedding and LLM processing can be time-consuming, especially for large documents or complex prompts.
-   **Dependency Management**: Keep `requirements.txt` up-to-date.
-   **Error Handling**: Robust error handling is important, especially for network requests, file operations, and API interactions.

## Tool Usage Patterns
-   **CLI for Operations**: All primary agent functions are exposed via `python -m app.main <command>`.
-   **Shell Scripts for Convenience**: `add_docs.sh` and `runScheduleExample` simplify common command invocations.
-   **Logging**: Standard Python `logging` module is used for outputting information about agent operations.
