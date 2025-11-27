"""
Simple test to verify embeddings are working
"""

from sentence_transformers import SentenceTransformer
from config import config


def test_embeddings():
    print("🧪 Testing embedding model loading...")

    try:
        # Load model (this might take a moment the first time)
        print(f"📥 Loading model: {config.EMBEDDING_MODEL}")
        model = SentenceTransformer(config.EMBEDDING_MODEL)

        # Test with simple text
        test_texts = [
            "This is a test document about artificial intelligence.",
            "Python is a programming language used for data science.",
            "Machine learning helps computers learn from data.",
        ]

        print("🔄 Generating embeddings...")
        embeddings = model.encode(test_texts)

        print(f"✅ Success!")
        print(f"   Model: {config.EMBEDDING_MODEL}")
        print(f"   Number of texts: {len(test_texts)}")
        print(f"   Embedding shape: {embeddings.shape}")
        print(f"   Vector dimension: {embeddings.shape[1]}")

        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        return False


if __name__ == "__main__":
    success = test_embeddings()
    if success:
        print("\n🎉 Embeddings test passed! Ready for next step.")
    else:
        print("\n⚠️  Embeddings test failed. Check your installation.")
