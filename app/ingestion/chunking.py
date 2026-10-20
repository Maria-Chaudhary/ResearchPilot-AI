
import re


def chunk_pages(
    pages: list[dict],
    chunk_size: int = 1800,
    overlap: int = 250,
) -> list[dict]:
    """
    Split extracted PDF pages into overlapping text chunks.

    Preserves page numbers and prefers paragraph, sentence,
    and word boundaries when creating chunks.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "overlap must be non-negative and smaller than chunk_size."
        )

    chunks = []
    step = chunk_size - overlap

    for page_data in pages:
        if "page" not in page_data or "text" not in page_data:
            raise ValueError(
                "Each page must contain 'page' and 'text' fields."
            )

        page_number = page_data["page"]
        text = page_data["text"]

        if not isinstance(text, str):
            raise ValueError("Page text must be a string.")

        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text).strip()

        if not text:
            continue

        start = 0
        text_length = len(text)

        while start < text_length:
            target_end = min(start + chunk_size, text_length)
            end = target_end

            if target_end < text_length:
                # Prefer a paragraph or line boundary.
                search_start = start + int(chunk_size * 0.70)

                paragraph_boundary = text.rfind(
                    "\n\n", search_start, target_end
                )
                line_boundary = text.rfind(
                    "\n", search_start, target_end
                )

                # Prefer the latest suitable paragraph/line boundary.
                boundary = max(paragraph_boundary, line_boundary)

                if boundary >= search_start:
                    end = boundary
                else:
                    # Otherwise, prefer a sentence boundary.
                    sentence_matches = list(
                        re.finditer(r"[.!?](?:[\"')\]]*)\s+", text[search_start:target_end])
                    )

                    if sentence_matches:
                        end = (
                            search_start
                            + sentence_matches[-1].end()
                        )
                    else:
                        # Last fallback: avoid splitting a word.
                        word_boundary = text.rfind(
                            " ", search_start, target_end
                        )

                        if word_boundary > start:
                            end = word_boundary

            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append({
                    "chunk_id": len(chunks),
                    "page": page_number,
                    "text": chunk_text,
                })

            if end >= text_length:
                break

            # Advance while retaining overlap between chunks.
            next_start = max(start + 1, end - overlap)

            # Avoid starting the next chunk in the middle of a word.
            while (
                next_start < end
                and next_start < text_length
                and not text[next_start - 1].isspace()
                and not text[next_start].isspace()
            ):
                next_start += 1

            # Safety check to guarantee progress.
            start = max(next_start, start + step)

    return chunks
