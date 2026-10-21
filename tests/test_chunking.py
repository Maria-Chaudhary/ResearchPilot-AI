
from app.ingestion.chunking import chunk_pages, split_into_passages


def test_empty_pages_return_no_chunks():
    assert chunk_pages([]) == []


def test_chunk_pages_preserves_page_number():
    pages = [
        {
            "page": 3,
            "text": "Introduction\n\nThis research studies machine learning.",
        }
    ]

    chunks = chunk_pages(pages)

    assert len(chunks) >= 1
    assert all(chunk["page"] == 3 for chunk in chunks)


def test_chunk_sections_are_tracked():
    pages = [
        {
            "page": 1,
            "text": (
                "Abstract\n\n"
                "This paper studies AI.\n\n"
                "Introduction\n\n"
                "The research question concerns AI systems."
            ),
        }
    ]

    chunks = chunk_pages(pages)

    abstract_chunks = [
        chunk for chunk in chunks
        if "This paper studies AI." in chunk["text"]
    ]
    introduction_chunks = [
        chunk for chunk in chunks
        if "The research question concerns AI systems." in chunk["text"]
    ]

    assert abstract_chunks
    assert introduction_chunks
    assert abstract_chunks[0]["section"].lower() == "abstract"
    assert introduction_chunks[0]["section"].lower() == "introduction"


def test_long_text_is_split_into_bounded_passages():
    text = " ".join(["research"] * 500)

    passages = split_into_passages(
        text,
        chunk_size=200,
        overlap=30,
    )

    assert len(passages) > 1
    assert all(len(passage) <= 200 for passage in passages)


def test_invalid_overlap_raises_error():
    pages = [{"page": 1, "text": "Some research text."}]

    try:
        chunk_pages(pages, chunk_size=100, overlap=100)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for invalid overlap")
