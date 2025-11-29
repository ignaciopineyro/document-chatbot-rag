import sys

from pathlib import Path
from src.document_processor import DocumentProcessor
from src.vector_store import VectorStore
from src.llm_client import LLMClient
from config import config

sys.path.append(str(Path(__file__).parent.parent))


class RAGPipeline:
    def __init__(self):
        self.document_processor = DocumentProcessor()
        self.vector_store = VectorStore()
        self.llm_client = LLMClient()

    def load_document(self, file_path: str) -> bool:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()

            chunks = self.document_processor.chunk_text(text)

            metadata = [
                {"source": file_path, "chunk_id": i, "chunk_size": len(chunk)}
                for i, chunk in enumerate(chunks)
            ]

            success = self.vector_store.add_documents(chunks, metadata)
            if success:
                print(f"Loaded {len(chunks)} chunks from {file_path}")
            return success

        except Exception as e:
            print(f"Error loading document: {e}")
            return False

    def query(self, question: str, top_k: int = 3) -> str:
        try:
            search_results = self.vector_store.search(question, limit=top_k)

            if not search_results:
                return "I don't have information to answer this question based on the loaded documents."

            context = [result["text"] for result in search_results]

            response = self.llm_client.generate_response(question, context)

            return response

        except Exception as e:
            return f"Error processing query: {e}"

    def get_status(self) -> dict:
        return {
            "llm_available": self.llm_client.test_connection(),
            "vector_store_ready": True,
            "model": config.OLLAMA_MODEL,
        }
