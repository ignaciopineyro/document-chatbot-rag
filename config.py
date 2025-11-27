"""
Configuration settings for the RAG chatbot MVP
"""

from pathlib import Path
from dataclasses import dataclass


@dataclass
class Config:
    # Vector Database
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    COLLECTION_NAME: str = "documents"

    # Embeddings
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    VECTOR_SIZE: int = 384  # all-MiniLM-L6-v2 vector size

    # LLM
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3.2"  # Updated to use llama3.2

    # Document processing
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50

    # Paths
    PROJECT_ROOT: Path = Path(__file__).parent
    DOCUMENTS_PATH: Path = PROJECT_ROOT / "data" / "documents"


# Global config instance
config = Config()

# Create necessary directories
config.DOCUMENTS_PATH.mkdir(parents=True, exist_ok=True)
