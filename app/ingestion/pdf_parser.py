
from pathlib import Path
import pymupdf


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
            text = page.get_text("text").strip()

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
