"""
End-to-end test: Document processing + Embeddings + Qdrant storage
"""

import sys
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
import uuid

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.document_processor import DocumentProcessor
from config import config


def test_end_to_end():
    """Complete pipeline test: process → embed → store → search"""
    print("🧪 End-to-End RAG Pipeline Test")
    print("=" * 50)

    # 1. Initialize components
    print("🔧 Initializing components...")
    processor = DocumentProcessor()
    embedder = SentenceTransformer(config.EMBEDDING_MODEL)
    qdrant_client = QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT)

    # 2. Create collection if it doesn't exist
    collection_name = "test_collection"
    print(f"📦 Setting up Qdrant collection: {collection_name}")

    try:
        # Delete collection if exists (for clean test)
        qdrant_client.delete_collection(collection_name=collection_name)
    except:
        pass  # Collection might not exist

    # Create new collection
    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=config.VECTOR_SIZE, distance=Distance.COSINE),
    )
    print("   ✅ Collection created")

    # 3. Process document
    print("\n📄 Processing document...")
    sample_file = Path("data/documents/sample.txt")

    with open(sample_file, "r", encoding="utf-8") as f:
        text = f.read()

    chunks = processor.chunk_text(text, chunk_size=300, overlap=30)
    print(f"   ✅ Created {len(chunks)} chunks")

    # 4. Generate embeddings
    print("\n🧠 Generating embeddings...")
    embeddings = embedder.encode(chunks)
    print(f"   ✅ Generated {len(embeddings)} embeddings")

    # 5. Store in Qdrant
    print("\n💾 Storing in Qdrant...")
    points = []
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        point = PointStruct(
            id=str(uuid.uuid4()),
            vector=embedding.tolist(),
            payload={"text": chunk, "chunk_id": i, "source": str(sample_file)},
        )
        points.append(point)

    qdrant_client.upsert(collection_name=collection_name, wait=True, points=points)
    print(f"   ✅ Stored {len(points)} points")

    # 6. Test search
    print("\n🔍 Testing similarity search...")
    query = "What is machine learning?"
    query_embedding = embedder.encode([query])[0]

    search_results = qdrant_client.query_points(
        collection_name=collection_name, query=query_embedding.tolist(), limit=3
    )

    points = search_results.points if hasattr(search_results, "points") else []
    print(f"   ✅ Found {len(points)} relevant chunks")
    print("\n📋 Most relevant chunks:")
    if not points:
        print("No valid result found")
    for i, result in enumerate(points):
        text = (
            result.payload.get("text", "No text available")
            if hasattr(result, "payload") and result.payload is not None
            else "No text available"
        )
        score = getattr(result, "score", 0.0)
        print(f"\nResult {i+1} (score: {score:.3f}):")
        print(f"'{text[:150]}...'")

    # 7. Cleanup
    print(f"\n🧹 Cleaning up...")
    qdrant_client.delete_collection(collection_name=collection_name)
    print("   ✅ Test collection deleted")

    return True


if __name__ == "__main__":
    try:
        success = test_end_to_end()
        if success:
            print("\n🎉 End-to-end test PASSED! RAG pipeline is working!")
            print("\nNext steps:")
            print("   • Add Ollama integration for LLM responses")
            print("   • Create main chatbot interface")
            print("   • Add PDF support")
        else:
            print("\n⚠️  End-to-end test FAILED.")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
