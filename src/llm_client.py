import ollama
from typing import List, Dict, Optional
from config import config


class LLMClient:
    """Client for Ollama LLM integration."""

    def __init__(self, model: Optional[str] = None):
        self.model = model or config.OLLAMA_MODEL

    def generate_response(
        self, prompt: str, context: Optional[List[str]] = None
    ) -> str:
        """Generate response using RAG context."""
        if context:
            context_text = "\n\n".join(context)
            full_prompt = f"""
            Based on the following context, please answer the question accurately:

            Context:
            {context_text}

            Question: {prompt}

            Answer:
            """
        else:
            full_prompt = prompt

        try:
            response = ollama.generate(
                model=self.model,
                prompt=full_prompt,
                options={"temperature": 0.1, "top_p": 0.9, "max_tokens": 500},
            )
            return response["response"].strip()
        except Exception as e:
            return f"Error generating response: {e}"

    def test_connection(self) -> bool:
        try:
            response = ollama.generate(model=self.model, prompt="Hello")
            return True
        except Exception:
            return False
