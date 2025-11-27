"""
Simple document processor for PDF files
"""

from typing import List, Optional, Union
from pathlib import Path
import PyPDF2
from config import config


class DocumentProcessor:
    """Handles PDF loading and text chunking."""

    def load_pdf(self, file_path: Union[str, Path]) -> str:
        """Load text from a PDF file."""
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")

        text = ""
        try:
            with open(file_path, "rb") as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    if page_text.strip():  # Only add non-empty pages
                        text += f"\n--- Page {page_num + 1} ---\n"
                        text += page_text + "\n"
        except Exception as e:
            raise RuntimeError(f"Error reading PDF {file_path}: {e}")

        return text.strip()

    def chunk_text(
        self, text: str, chunk_size: Optional[int] = None, overlap: Optional[int] = None
    ) -> List[str]:
        """Split text into chunks for vector storage."""
        chunk_size = chunk_size or config.CHUNK_SIZE
        overlap = overlap or config.CHUNK_OVERLAP

        if len(text) <= chunk_size:
            return [text]

        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size

            # If we're not at the end, try to break at a sentence or word boundary
            if end < len(text):
                # Look for sentence ending
                sentence_end = text.rfind(".", start, end)
                if sentence_end > start + chunk_size // 2:
                    end = sentence_end + 1
                else:
                    # Look for word boundary
                    word_end = text.rfind(" ", start, end)
                    if word_end > start + chunk_size // 2:
                        end = word_end

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            # Move start position with overlap
            start = end - overlap
            if start <= 0:
                start = end

        return chunks

    def process_document(self, file_path: str) -> tuple[str, List[str]]:
        """Complete document processing pipeline."""
        print(f"📄 Loading document: {file_path}")

        # Load PDF
        text = self.load_pdf(file_path)
        print(f"   Text length: {len(text)} characters")

        # Chunk text
        chunks = self.chunk_text(text)
        print(f"   Created {len(chunks)} chunks")

        return text, chunks
