
import pymupdf
import pytest

from app.ingestion.pdf_parser import extract_text_from_pdf


def test_extract_text_and_page_numbers(tmp_path):
    pdf_path = tmp_path / "research_paper.pdf"

    with pymupdf.open() as document:
        page1 = document.new_page()
        page1.insert_text((72, 72), "Abstract: AI research findings.")

        page2 = document.new_page()
        page2.insert_text((72, 72), "Conclusion: Further research is needed.")

        document.save(str(pdf_path))

    pages = extract_text_from_pdf(str(pdf_path))

    assert len(pages) == 2
    assert pages[0]["page"] == 1
    assert "AI research findings" in pages[0]["text"]
    assert pages[1]["page"] == 2
    assert "Further research is needed" in pages[1]["text"]


def test_missing_pdf_raises_error(tmp_path):
    missing_path = tmp_path / "missing.pdf"

    with pytest.raises(FileNotFoundError):
        extract_text_from_pdf(str(missing_path))


def test_non_pdf_file_raises_error(tmp_path):
    text_path = tmp_path / "notes.txt"
    text_path.write_text("Not a PDF", encoding="utf-8")

    with pytest.raises(ValueError, match="must be a PDF"):
        extract_text_from_pdf(str(text_path))


def test_empty_pdf_raises_error(tmp_path):
    pdf_path = tmp_path / "empty.pdf"

    with pymupdf.open() as document:
        document.new_page()
        document.save(str(pdf_path))

    with pytest.raises(ValueError, match="No readable text"):
        extract_text_from_pdf(str(pdf_path))
