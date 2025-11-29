import sys

from pathlib import Path
from src.rag_pipeline import RAGPipeline

sys.path.append(str(Path(__file__).parent / "src"))


def main():
    print("RAG Document Chatbot")
    print("-" * 40)

    rag = RAGPipeline()

    status = rag.get_status()
    if not status["llm_available"]:
        print("Error: Ollama LLM not available. Please check your installation.")
        return

    print(f"Model: {status['model']}")
    print("System ready.")

    sample_doc = Path("data/documents/sample.txt")
    if sample_doc.exists():
        print(f"Loading document: {sample_doc}")
        success = rag.load_document(str(sample_doc))
        if not success:
            print("Warning: Failed to load document")
    else:
        print("No sample document found. You can still ask general questions.")

    print("\nYou can now ask questions! Type 'quit' to exit.")
    print("-" * 40)

    while True:
        try:
            question = input("\nYou: ").strip()

            if question.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break

            if not question:
                continue

            print("\nThinking...")
            response = rag.query(question)
            print(f"\nBot: {response}")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
