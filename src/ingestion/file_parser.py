import os
from typing import Optional
import fitz  # PyMuPDF
import docx  # python-docx


class FileParser:

    SUPPORTED_EXTENSIONS = {".pdf", ".docx"}

    def parse(self, file_path: str) -> str:
        # Check file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        # Get extension
        _, extension = os.path.splitext(file_path)
        extension = extension.lower()

        if extension == ".pdf":
            return self._parse_pdf(file_path)
        elif extension == ".docx":
            return self._parse_docx(file_path)
        else:
            raise ValueError(
                f"Unsupported file type: {extension}. "
                f"Supported: {self.SUPPORTED_EXTENSIONS}"
            )

    def _parse_pdf(self, file_path: str) -> str:
        text_parts = []
        with fitz.open(file_path) as pdf_document:
            for page_number in range(len(pdf_document)):
                page = pdf_document[page_number]
                page_text = page.get_text()
                if page_text.strip():
                    text_parts.append(
                        f"\n[Page {page_number + 1}]\n{page_text}"
                    )

        full_text = "\n".join(text_parts)
        if not full_text.strip():
            raise ValueError("No text could be extracted from the PDF.")
        return full_text

    def _parse_docx(self, file_path: str) -> str:
        doc = docx.Document(file_path)
        text_parts = []
        for paragraph in doc.paragraphs:
            if paragraph.text.strip():
                text_parts.append(paragraph.text)

        full_text = "\n".join(text_parts)
        if not full_text.strip():
            raise ValueError("No text could be extracted from the DOCX.")
        return full_text