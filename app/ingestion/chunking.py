
import re


RESEARCH_HEADINGS = {
    "abstract",
    "introduction",
    "background",
    "related work",
    "literature review",
    "research methodology",
    "methodology",
    "materials and methods",
    "methods",
    "experimental setup",
    "results",
    "results and discussion",
    "discussion",
    "limitations",
    "future work",
    "conclusion",
    "conclusions",
    "references",
}


def clean_text(text: str) -> str:
    """Normalize text extracted from research papers."""
    text = text.replace("\u00ad", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def is_research_heading(line: str) -> bool:
    """Recognize common academic section headings."""
    normalized = re.sub(
        r"^\d+(?:\.\d+)*\.?\s*",
        "",
        line.strip(),
    )
    normalized = normalized.strip(" .:-").lower()

    return normalized in RESEARCH_HEADINGS


def split_into_passages(
    text: str,
    chunk_size: int,
    overlap: int,
) -> list[str]:
    """Split text while preserving paragraphs where possible."""
    paragraphs = [
        paragraph.strip()
        for paragraph in re.split(r"\n\s*\n", text)
        if paragraph.strip()
    ]

    chunks = []
    current = ""

    for paragraph in paragraphs:
        if len(paragraph) > chunk_size:
            if current:
                chunks.append(current)
                current = ""

            start = 0

            while start < len(paragraph):
                end = min(start + chunk_size, len(paragraph))

                if end < len(paragraph):
                    boundary = paragraph.rfind(" ", start, end)
                    if boundary > start:
                        end = boundary

                passage = paragraph[start:end].strip()

                if passage:
                    chunks.append(passage)

                if end >= len(paragraph):
                    break

                start = max(start + 1, end - overlap)

            continue

        candidate = (
            f"{current}\n\n{paragraph}".strip()
            if current
            else paragraph
        )

        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                chunks.append(current)

            current = paragraph

    if current:
        chunks.append(current)

    return chunks


def chunk_pages(
    pages: list[dict],
    chunk_size: int = 1200,
    overlap: int = 200,
) -> list[dict]:
    """Create chunks with page numbers and research-section metadata."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "overlap must be non-negative and smaller than chunk_size."
        )

    chunks = []

    for page_data in pages:
        if not isinstance(page_data, dict):
            raise ValueError("Each page must be a dictionary.")

        if "page" not in page_data or "text" not in page_data:
            raise ValueError(
                "Each page must contain 'page' and 'text' fields."
            )

        page_number = page_data["page"]
        raw_text = page_data["text"]

        if not isinstance(raw_text, str):
            raise ValueError("Page text must be a string.")

        text = clean_text(raw_text)

        if not text:
            continue

        current_section = "General"
        section_lines = []

        def save_section(lines, section_name):
            section_text = "\n".join(lines).strip()

            if not section_text:
                return

            passages = split_into_passages(
                section_text,
                chunk_size,
                overlap,
            )

            for passage in passages:
                chunks.append({
                    "chunk_id": len(chunks),
                    "page": page_number,
                    "section": section_name,
                    "text": passage,
                })

        for line in text.splitlines():
            if is_research_heading(line):
                save_section(section_lines, current_section)
                section_lines = []
                current_section = line.strip()
            else:
                section_lines.append(line)

        save_section(section_lines, current_section)

    return chunks
