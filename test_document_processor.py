"""
Test document processor with a simple text file
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.document_processor import DocumentProcessor


def create_sample_text_file():
    """Create a simple text file for testing (since we don't have PDF yet)"""
    sample_text = """
This is a sample document for testing our RAG system.

Artificial Intelligence is a broad field that encompasses machine learning, deep learning, 
and natural language processing. It aims to create intelligent systems that can perform 
tasks that typically require human intelligence.

Machine Learning is a subset of AI that enables computers to learn and improve from 
experience without being explicitly programmed. It uses statistical techniques to give 
computers the ability to "learn" from data.

Natural Language Processing (NLP) is another important area of AI that focuses on the 
interaction between computers and human language. It involves developing algorithms and 
models that can understand, interpret, and generate human language.

Deep Learning is a subset of machine learning that uses neural networks with multiple 
layers (hence "deep") to model and understand complex patterns in data. It has been 
particularly successful in areas like image recognition and language processing.
"""

    sample_file = Path("data/documents/sample.txt")
    sample_file.parent.mkdir(parents=True, exist_ok=True)

    with open(sample_file, "w", encoding="utf-8") as f:
        f.write(sample_text)

    return sample_file


def test_document_processor():
    """Test the document processor with chunking"""
    print("🧪 Testing Document Processor")
    print("-" * 40)

    # Create sample file
    sample_file = create_sample_text_file()
    print(f"📝 Created sample file: {sample_file}")

    # Test text chunking (since we don't have PDF yet)
    processor = DocumentProcessor()

    # Read the text file directly
    with open(sample_file, "r", encoding="utf-8") as f:
        text = f.read()

    print(f"📄 Original text length: {len(text)} characters")

    # Test chunking
    chunks = processor.chunk_text(text, chunk_size=200, overlap=20)

    print(f"✅ Created {len(chunks)} chunks")
    print("\n📋 First few chunks:")
    for i, chunk in enumerate(chunks[:3]):
        print(f"\nChunk {i+1} ({len(chunk)} chars):")
        print(f"'{chunk[:100]}...'")

    return True


if __name__ == "__main__":
    success = test_document_processor()
    if success:
        print("\n🎉 Document processor test passed!")
    else:
        print("\n⚠️  Document processor test failed.")
