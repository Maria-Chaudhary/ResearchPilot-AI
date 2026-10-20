
from pathlib import Path
import pymupdf
import re
import unicodedata


def extract_text_from_pdf(pdf_path: str) -> list[dict]:
    """Extract text from a PDF while preserving page numbers."""

    path = Path(pdf_path)

    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {path}")

    if path.suffix.lower() != ".pdf":
        raise ValueError("The supplied file must be a PDF.")

    pages = []

    with pymupdf.open(str(path)) as document:
        if document.needs_pass:
            raise ValueError("This PDF requires a password.")

        for page_number, page in enumerate(document, start=1):
            # Extract text from the current page.
            text = page.get_text("text")

            # Normalize unusual Unicode characters.
            text = unicodedata.normalize("NFKC", text)

            # Replace incorrectly decoded PDF bullet symbols.
            text = text.replace("\uf0b2", "-")
            text = text.replace("", "-")

            # Remove unwanted control characters while preserving
            # newlines and tabs.
            text = "".join(
                char
                for char in text
                if char in "\n\t"
                or unicodedata.category(char) != "Cc"
            )

            # Clean excess whitespace.
            text = re.sub(r"[ \t]+", " ", text)
            text = re.sub(r"\n{3,}", "\n\n", text)
            text = text.strip()

            # Save the page if it contains readable text.
            if text:
                pages.append({
                    "page": page_number,
                    "text": text,
                })

    if not pages:
        raise ValueError(
            "No readable text found. The PDF may be scanned or empty."
        )

    return pages
