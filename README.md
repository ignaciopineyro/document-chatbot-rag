# document-chatbot-rag

A nerdy chatbot for querying your own documents using open-source models and RAG. The system allows uploading text files, indexing them in a vector database, and querying them using natural language.

## How it works

1. The user uploads text documents.
2. The documents are chunked, embedded, and stored in a local vector database.
3. When the user asks a question:
    - The question is embedded
    - Relevant chunks are retrieved from the vector store
    - The selected context is passed to the model

4. The model generates the final answer based on the retrieved content.

## Tech stack

| Tool                                | Purpose                                    |
| ----------------------------------- | ------------------------------------------ |
| LangChain                           | RAG orchestration and chatbot pipeline     |
| Qdrant                      | Vector database for semantic search        |
| Sentence Transformers / HuggingFace | Embeddings generation                      |
| Ollama                              | Local open-source models                   |
| FastAPI                             | Main Framework                      |
| Docker                   | Run everything on a container |

## Project Structure (MVP)

```
document-chatbot-rag/
├── src/
│   ├── __init__.py
│   ├── main.py                # Main entry point
│   ├── config.py              # System configuration
│   ├── document_processor.py  # Document processing and chunking
│   ├── embeddings.py          # Embedding generation
│   ├── vector_store.py        # Qdrant management
│   ├── retriever.py           # Relevant chunk retrieval
│   ├── llm_client.py          # Ollama client
│   └── chatbot.py             # Complete RAG pipeline
├── data/
│   └── documents/             # Documents
├── requirements.txt
├── docker-compose.yml         
└── README.md
```

## MVP Flow

1. Initialization (`main.py`):
    - Load hardcoded document from data/documents/
    - Process and save to vector store if not exists
2. Processing (`document_processor.py`):
    - Extract text from PDF
    - Split into chunks with overlap
3. Embeddings (`embeddings.py`):
    - Use local Sentence Transformers
    - Generate embeddings for chunks
4. Vector Store (`vector_store.py`):
    - Connect to Qdrant (Docker)
    - Store and search chunks
5. Chatbot (`chatbot.py`):
    - Receive user question
    - Retrieve relevant chunks
    - Generate response with Ollama