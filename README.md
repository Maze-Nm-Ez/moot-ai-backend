# Chatbot Backend by NonePy

## Overview

This project is the backend for an advanced chatbot application. It utilizes LangGraph to create a sophisticated agentic workflow for processing user queries, retrieving relevant information from various documents, generating AI responses, and evaluating the interaction quality. The system is designed to work with local documents (.txt and .pdf) which are automatically processed and indexed for retrieval.

## Prerequisites

- Python 3.12 or higher
- Poetry for dependency management

## Setup

1.  **Clone the repository:**

    ```bash
    git clone <your-repository-url>
    cd moot-ai-backend
    ```

2.  **Create a `.env` file:**
    Create a file named `.env` in the root of the project and add the following environment variables:

    ```env
    OPENAI_API_KEY="your_openai_api_key_here"
    # TAVILY_API_KEY="your_tavily_api_key_here" # Optional: For web search capabilities
    ```

    - `OPENAI_API_KEY`: Essential for text embeddings and AI response generation.
    - `TAVILY_API_KEY`: (Optional) Required if you intend to use or enable web search functionalities via Tavily.

3.  **Install dependencies:**
    Use Poetry to install the required packages:

    ```bash
    poetry install
    ```

4.  **Prepare Documents:**
    - Create a directory named `documents` in the root of the project:
      ```bash
      mkdir documents
      ```
    - Place your `.txt` and/or `.pdf` files into this `documents/` directory. These files will be automatically processed by the application at startup.

## Document Processing

Upon startup, the application automatically:

1.  Scans the `documents/` directory for `.txt` and `.pdf` files.
2.  Loads the content from these files.
3.  Splits the documents into manageable chunks.
4.  Generates embeddings for these chunks using OpenAI.
5.  Stores these embeddings in a local Chroma vector store located at `./vector_stores/main_document_vectorstore`.
6.  Creates Langchain retrievers for each document source (derived from the filename). These retrievers are accessible via the `document_retrievers` dictionary in `RAG.tools.vectore_store_retriever`.

This process ensures that your documents are ready for efficient information retrieval as part of the chatbot's workflow.

## Running the Application

To run the FastAPI application in development mode:

```bash
poetry run fastapi dev main.py
```

This will typically start the server on `http://127.0.0.1:8000`. The `main.py` file is assumed to be the entry point for your FastAPI application.

## Project Structure

- `main.py`: (Assumed) The main FastAPI application file.
- `pyproject.toml`: Defines project dependencies and metadata for Poetry.
- `.env`: (User-created) Stores environment variables like API keys.
- `nodes/`: Contains Python modules defining individual nodes (processing steps) for the LangGraph workflow.
- `RAG/`: Contains modules related to the Retrieval Augmented Generation capabilities.
  - `RAG/graph.py`: Defines the LangGraph state and workflow.
  - `RAG/graph_state.py`: Defines the state object for the graph.
  - `RAG/tools/`: Contains tools used within the graph, such as `vectore_store_retriever.py`.
- `documents/`: (User-created) Directory where you place your `.txt` and `.pdf` files for processing.
- `vector_stores/`: Directory where the Chroma vector store (e.g., `main_document_vectorstore`) is persisted.

## Notes

- If you add new documents to the `documents/` folder, you will need to restart the server for them to be processed and included in the vector store.
