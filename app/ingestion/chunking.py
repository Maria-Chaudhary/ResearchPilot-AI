
def chunk_pages(
    pages: list[dict],
    chunk_size: int = 1800,
    overlap: int = 250,
) -> list[dict]:
    """Split PDF pages into chunks while preserving page numbers."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "overlap must be non-negative and smaller than chunk_size."
        )

    chunks = []

    for page_data in pages:
        page_number = page_data["page"]
        text = page_data["text"].strip()

        if not text:
            continue

        start = 0

        while start < len(text):
            end = min(start + chunk_size, len(text))

            if end < len(text):
                boundary = max(
                    text.rfind(" ", start + int(chunk_size * 0.7), end),
                    text.rfind("\n", start + int(chunk_size * 0.7), end),
                )

                if boundary > start:
                    end = boundary

            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append({
                    "chunk_id": len(chunks),
                    "page": page_number,
                    "text": chunk_text,
                })

            if end >= len(text):
                break

            start = end - overlap

    return chunks
