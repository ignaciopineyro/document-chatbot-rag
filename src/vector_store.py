import uuid

from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from config import config


class VectorStore:
    def __init__(self):
        self.client = QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT)
        self.embedder = SentenceTransformer(config.EMBEDDING_MODEL)
        self.collection_name = config.COLLECTION_NAME
        self._ensure_collection()

    def _ensure_collection(self):
        try:
            collections = self.client.get_collections().collections
            if not any(c.name == self.collection_name for c in collections):
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=config.VECTOR_SIZE, distance=Distance.COSINE
                    ),
                )
        except Exception as e:
            print(f"Error ensuring collection: {e}")

    def add_documents(
        self, texts: List[str], metadata: Optional[List[Dict]] = None
    ) -> bool:
        try:
            embeddings = self.embedder.encode(texts)
            points = []

            for i, (text, embedding) in enumerate(zip(texts, embeddings)):
                point_metadata = metadata[i] if metadata and i < len(metadata) else {}
                point_metadata["text"] = text

                point = PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding.tolist(),
                    payload=point_metadata,
                )
                points.append(point)

            self.client.upsert(
                collection_name=self.collection_name, wait=True, points=points
            )
            return True
        except Exception as e:
            print(f"Error adding documents: {e}")
            return False

    def search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        try:
            query_embedding = self.embedder.encode([query])[0]

            search_results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_embedding.tolist(),
                limit=limit,
            )

            results = []
            points = getattr(search_results, "points", [])

            for point in points:
                if point and hasattr(point, "payload") and point.payload:
                    results.append(
                        {
                            "text": point.payload.get("text", ""),
                            "score": getattr(point, "score", 0.0),
                            "metadata": {
                                k: v for k, v in point.payload.items() if k != "text"
                            },
                        }
                    )

            return results
        except Exception as e:
            print(f"Error searching: {e}")
            return []

    def clear_collection(self) -> bool:
        try:
            self.client.delete_collection(collection_name=self.collection_name)
            self._ensure_collection()
            return True
        except Exception as e:
            print(f"Error clearing collection: {e}")
            return False
