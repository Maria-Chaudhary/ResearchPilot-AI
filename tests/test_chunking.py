import re

import pytest

from app.ingestion import chunking as c


def test_public_api_preserved():
    for name in ("RESEARCH_HEADINGS", "clean_text", "normalize_heading",
                 "is_research_heading", "split_into_passages", "chunk_pages"):
        assert hasattr(c, name)


def test_heading_detection():
    assert c.is_research_heading("1. INTRODUCTION")
    assert c.is_research_heading("V. FINDINGS, CONCLUSION AND RECOMMENDATIONS")
    assert c.is_research_heading("C. OBJECTIVES OF THE STUDY")
    assert c.normalize_heading("2.1 Results.") == "results"


def test_overlap_is_whole_sentences_and_size_respected():
    text = " ".join(f"Sentence number {i} talks about retrieval augmented generation systems." for i in range(40))
    passages = c.split_into_passages(text, chunk_size=400, overlap=120)
    assert all(len(p) <= 400 for p in passages)
    for prev, nxt in zip(passages, passages[1:]):
        assert nxt[0].isupper()
        assert (nxt.split(". ")[0] + ".") in prev


def test_no_chunk_starts_or_ends_mid_sentence(structured_paper):
    for chunk in structured_paper:
        text = chunk["text"]
        assert not text[0].islower()
        if chunk["section"] not in ("References", "General"):  # title block has no full stop
            assert re.search(r'[.!?]["\')\]]*$', text), text[-40:]


def test_headers_removed_and_sections_carry_over():
    header = "International Journal of X\nwww.x.org | a@b.com\nVolume 5 Issue 3\nISSN: 1234-5678"
    pages = [{"page": i, "text": f"{header}\n" + body} for i, body in enumerate([
        "ABSTRACT\nWe study things and report findings in detail here.",
        "potential uses of the method in practice. More text follows here today.\n\nA. FINDINGS\n\nThe objective of the study was to know things. Another sentence appears here.",
        "The findings continue on this page with a complete sentence here.",
    ], start=1)]
    chunks = c.chunk_pages(pages, chunk_size=300, overlap=50)
    joined = " ".join(x["text"] for x in chunks)
    assert "ISSN" not in joined and "www" not in joined
    assert any(x["page"] == 3 and x["section"] == "Findings" for x in chunks)
    assert any("study things" in x["text"] or "uses of the method" in x["text"] for x in chunks)


def test_title_stays_text_and_ids_sequential(structured_paper):
    assert structured_paper[0]["section"] == "General"
    assert "ARTIFICIAL INTELLIGENCE ON PRODUCTIVITY" in structured_paper[0]["text"]
    assert [x["chunk_id"] for x in structured_paper] == list(range(len(structured_paper)))
    assert [x["section"].lower() for x in structured_paper if x["page"] == 12] == ["references"]


@pytest.mark.parametrize("kwargs", [{"chunk_size": 0}, {"overlap": -1}, {"chunk_size": 100, "overlap": 100}])
def test_invalid_sizes_raise(kwargs):
    with pytest.raises(ValueError):
        c.chunk_pages([{"page": 1, "text": "x"}], **kwargs)


@pytest.mark.parametrize("pages", [[1], [{"page": 1}], [{"page": 0, "text": "x"}], [{"page": 1, "text": 5}]])
def test_invalid_pages_raise(pages):
    with pytest.raises(ValueError):
        c.chunk_pages(pages)


def test_markdown_headings_from_pdf_extractor():
    """pymupdf4llm returns Markdown: '## Introduction', '## **Conclusion**'."""
    pages = [
        {"page": 1, "text": "# AI and Society\nMaria Khan\n\n## Introduction\nThis essay studies artificial intelligence and its goals in detail."},
        {"page": 2, "text": "**==> picture [200 x 100] intentionally omitted <==**\n\n## Creativity\nScientists believe computers may become creative in the future.\n\n**Table 1**\nValues here are shown for reference only."},
        {"page": 3, "text": "## **Conclusion**\nIn conclusion, artificial intelligence will keep growing and changing work.\n\n**References**\nSmith, J. 2019. Journal of Things. vol. 3 pp. 1-9."},
    ]
    chunks = c.chunk_pages(pages, chunk_size=300, overlap=50)
    sections = {(x["page"], x["section"]) for x in chunks}
    assert (1, "Introduction") in sections and (2, "Creativity") in sections
    assert (3, "Conclusion") in sections and (3, "References") in sections
    joined = " ".join(x["text"] for x in chunks)
    assert "picture" not in joined and "Table 1" in joined          # caption stays text, not a section
    assert all(x["section"] != "Table 1" for x in chunks)
    assert c.is_research_heading("## **Conclusion**") and c.normalize_heading("## 3.1 **Results**") == "results"