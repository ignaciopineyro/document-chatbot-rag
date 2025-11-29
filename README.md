# document-chatbot-rag

A RAG chatbot for querying your own documents using open-source models. Upload documents, ask questions, get answers based on your content using vector search and local LLMs.

## Quick Start

**Prerequisites:** Docker and Docker Compose only (no Python installation needed)

### Start the System
```bash
./start-services.sh
```

This single command will:
- Start Qdrant vector database service
- Start Ollama LLM service
- Install Python dependencies automatically
- Download llama3.2 model
- Launch interactive chatbot

### Stop the System
```bash
./stop-services.sh
```

### Add Your Documents
Place your text or PDF files in the `data/documents/` folder before starting the system.

### Example Interaction
```
You: What is artificial intelligence?
Bot: Based on the context, Artificial Intelligence (AI) is a broad field 
     that encompasses machine learning, deep learning, and natural language 
     processing, aiming to create intelligent systems that can perform 
     tasks typically requiring human intelligence.
```

**No local Python/pip installation required** - everything runs in Docker containers.


## How it works

1. The user uploads a text document -> "Machine learning helps computers learn from data"
2. This document is preprocessed (chunking) -> `chunks = ["Machine learning helps", "computers learn from data"]`
3. Chunks are embedded using transformers (SentenceTransformer) resulting in sentence vectors  -> `[[0.1, 0.8, ...], [0.2, 0.7, ...]]`
4. These vectors are stored in a Vector DB (Qdrant) labeled properly with the text chunk they represent -> `{vector: [0.1, 0.8, ...], text: "Machine learning helps"}`
5. The user asks something -> `"What is ML?"`
6. User input is embedded and the vector is stored in the Vector DB
7. Similarity search is done to get the best result according to a score -> `"Machine learning helps..." (score: 0.85)`

## Tech stack

| Tool                                | Purpose                                    |
| ----------------------------------- | ------------------------------------------ |
| LangChain                           | RAG orchestration and chatbot pipeline     |
| Qdrant                      | Vector database for semantic search        |
| Sentence Transformers / HuggingFace | Embeddings generation                      |
| Ollama                              | Local open-source models                   |
| FastAPI                             | Main Framework                      |
| Docker                   | Run everything on a container |


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