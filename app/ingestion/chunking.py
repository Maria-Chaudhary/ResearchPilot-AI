import re
from collections import Counter


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
    # --- added ---
    "objectives",
    "objectives of the study",
    "research objectives",
    "research questions",
    "statement of the problem",
    "problem statement",
    "significance of the study",
    "scope of the study",
    "theoretical framework",
    "conceptual framework",
    "research design",
    "data analysis",
    "proposed method",
    "proposed approach",
    "approach",
    "experiments",
    "experimental results",
    "evaluation",
    "analysis",
    "findings",
    "findings, conclusion and recommendations",
    "conclusion and recommendations",
    "recommendations",
    "acknowledgement",
    "acknowledgements",
    "acknowledgment",
    "acknowledgments",
    "bibliography",
    "concluding remarks",
    "summary and conclusion",
    "summary and conclusions",
    "discussion and conclusion",
    "conclusion and future work",
    "conclusions and future work",
    "threats to validity",
    "contributions",
    "summary",
    "overview",
    "dataset",
    "data collection",
    "implementation",
    "case study",
}

_SMALL_WORDS = {"and", "of", "the", "in", "for", "to", "a", "an", "on"}

_BULLET_LINE = re.compile(r"^\s*[-\u2022\u25aa\u25cf\u25e6\u2023*]\s+")

# Journal running headers, page footers, URLs, e-mails, ISSN, copyright ...
_BOILERPLATE_LINE = re.compile(
    r"^\s*(?:page\s+\d+(?:\s+of\s+\d+)?|\d{1,3}|https?://\S+|.*\bwww\.\S+.*|"
    r".*\S+@\S+\.\S+.*|issn\b.*|volume\s+\d+.*|vol\.\s*\d+.*|"
    r"international journal.*|copyright.*|\u00a9.*|all rights reserved.*)\s*$",
    re.IGNORECASE,
)

_ENUM_HEADING = re.compile(r"^(?:[A-Z]|[IVXLC]+|\d{1,2}(?:\.\d+)*)[.)]\s+[A-Z]")
_NUMBERED_HEADING = re.compile(r"^\d{1,2}(?:\.\d+)*\.?\s+[A-Z]")
_END_PUNCT = re.compile(r'[.!?]["\')\]]*$')


def clean_text(text: str) -> str:
    """Clean text extracted from research papers."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00ad", "")
    # Private-use / symbol bullets at the start of a line become "- ".
    text = re.sub(
        r"(?m)^[ \t]*[\uf000-\uf8ff\u2022\u25aa\u25cf\u25e6\u2023][ \t]*", "- ", text
    )
    text = text.replace("\uf0b2", "-")
    text = text.replace("\uf0b7", "-")
    # Re-join words hyphenated across a line break ("devel-\nopment").
    text = re.sub(r"(?<=[A-Za-z])-[ \t]*\n[ \t]*(?=[a-z])", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_heading(line: str) -> str:
    """Normalize numbered research headings."""
    line = line.strip()
    line = re.sub(r"^\d+(?:\.\d+)*\.?\s*", "", line)
    # "V. FINDINGS", "A. Findings", "c) Methods"
    line = re.sub(r"^(?:[IVXLCivxlc]+|[A-Za-z])[.)]\s+", "", line)
    return line.strip(" .:-").lower()


def is_research_heading(line: str) -> bool:
    """Check whether a line resembles a known research heading."""
    normalized = normalize_heading(line)

    if normalized in RESEARCH_HEADINGS:
        return True

    # Support numbered headings such as 1. INTRODUCTION.
    return False


def is_numbered_heading(line: str) -> bool:
    """Generic numbered sub-heading such as '3.1 Dataset collection'."""
    line = line.strip()

    return bool(
        _NUMBERED_HEADING.match(line)
        and len(line.split()) <= 10
        and not _END_PUNCT.search(line)
    )


def _is_caps_heading(line: str) -> bool:
    line = line.strip()

    return bool(
        line
        and line.isupper()
        and 2 <= len(line.split()) <= 12
        and len(re.findall(r"[A-Za-z]", line)) >= 6
        and not _END_PUNCT.search(line)
    )


def _pretty_title(text: str) -> str:
    words = text.split()
    out = []

    for i, word in enumerate(words):
        out.append(word if (i and word in _SMALL_WORDS) else word.capitalize())

    return " ".join(out)


def _heading_title(line: str):
    """Return a clean section title when the line is a heading, else None."""
    stripped = line.strip()

    if not stripped:
        return None

    if (
        is_research_heading(stripped)
        or _is_caps_heading(stripped)
        or is_numbered_heading(stripped)
    ):
        title = _pretty_title(normalize_heading(stripped))
        return title or None

    return None


# ---------------------------------------------------------
# Page-level cleanup
# ---------------------------------------------------------

def _norm_line(line: str) -> str:
    return re.sub(r"\d+", "#", line.strip().lower())


def _repeated_lines(page_texts: list[str], min_pages: int = 3, ratio: float = 0.4):
    """Short lines repeated on many pages are running headers/footers."""
    if len(page_texts) < min_pages:
        return set()

    counts: Counter = Counter()

    for text in page_texts:
        counts.update(
            {
                _norm_line(line)
                for line in text.splitlines()
                if 3 <= len(line.strip()) <= 120
            }
        )

    threshold = max(min_pages, ratio * len(page_texts))

    return {line for line, count in counts.items() if line and count >= threshold}


def remove_boilerplate_lines(text: str, repeated: set | None = None) -> str:
    """Drop page numbers, URLs, e-mails, ISSN/volume lines, running headers."""
    repeated = repeated or set()
    kept = []

    for line in text.splitlines():
        stripped = line.strip()

        if stripped and len(stripped) <= 140:
            if _BOILERPLATE_LINE.match(stripped):
                continue

            if _norm_line(stripped) in repeated and not is_research_heading(stripped):
                continue

        kept.append(line)

    return re.sub(r"\n{3,}", "\n\n", "\n".join(kept)).strip()


def _move_dangling_fragments(texts: list[str]) -> list[str]:
    """
    When a sentence is split by a page break, move the unfinished tail of
    page N to the start of page N+1 so no passage starts or ends mid-sentence.
    """
    texts = list(texts)

    for i in range(len(texts) - 1):
        current = texts[i].rstrip()
        following = texts[i + 1].lstrip()

        if not current or not following:
            continue

        if re.search(r'[.!?:]["\')\]]*$', current):
            continue

        if not following[0].islower():  # next page does not continue a sentence
            continue

        boundary = max(
            (m.end() for m in re.finditer(r"[.!?][\"')\]]*\s+|\n\s*\n", current)),
            default=0,
        )
        fragment = current[boundary:].strip()

        if not fragment or len(fragment) > 600:
            continue

        texts[i] = current[:boundary].rstrip()
        texts[i + 1] = f"{fragment} {following}"

    return texts


def _merge_wrapped_headings(lines: list[str]) -> list[str]:
    """'V. FINDINGS, CONCLUSION AND' + 'RECOMMENDATIONS' -> one heading line."""
    merged: list[str] = []

    for line in lines:
        stripped = line.strip()

        if (
            merged
            and stripped
            and _is_caps_heading_like(stripped)
            and _is_caps_heading_like(merged[-1])
            and not is_research_heading(merged[-1])
        ):
            merged[-1] = merged[-1].strip() + " " + stripped
        else:
            merged.append(line)

    return merged


def _is_caps_heading_like(line: str) -> bool:
    line = line.strip()

    return bool(
        line
        and line.isupper()
        and len(line.split()) <= 12
        and not _END_PUNCT.search(line)
    )


# ---------------------------------------------------------
# Sentence-aware splitting
# ---------------------------------------------------------

_ABBREV_END = re.compile(
    r"(?:\b(?:Fig|Figs|Eq|Eqs|Tab|Sec|al|vs|etc|cf|approx|No|Dr|Prof)|\be\.g|\bi\.e)\.$",
    re.IGNORECASE,
)

_SENTENCE_END = re.compile(r'[.!?]["\')\]]*\s+(?=[A-Z0-9\[(\u201c"\'])')


def _split_sentences(block: str) -> list[str]:
    sentences, start = [], 0

    for match in _SENTENCE_END.finditer(block):
        end = match.start() + len(match.group(0).rstrip())
        sentences.append(block[start:end].strip())
        start = match.end()

    tail = block[start:].strip()

    if tail:
        sentences.append(tail)

    merged: list[str] = []

    for sentence in sentences:
        if merged and _ABBREV_END.search(merged[-1]):
            merged[-1] += " " + sentence
        else:
            merged.append(sentence)

    return [s for s in merged if s]


def _split_long_sentence(sentence: str, limit: int) -> list[str]:
    """Only for sentences longer than chunk_size: cut at clauses, never mid-word."""
    pieces, rest = [], sentence

    while len(rest) > limit:
        cut = -1

        for sep in ("; ", ": ", ", "):
            index = rest.rfind(sep, 0, limit)

            if index > limit * 0.4:
                cut = max(cut, index + len(sep.strip()))

        if cut < 0:
            cut = rest.rfind(" ", 0, limit)

            if cut <= 0:
                cut = limit

        pieces.append(rest[:cut].strip())
        rest = rest[cut:].strip()

    if rest:
        pieces.append(rest)

    return pieces


def _units(text: str, chunk_size: int) -> list[tuple[str, str]]:
    """(sentence, separator-before) units. Bullets stay as their own items."""
    units: list[tuple[str, str]] = []

    for p_index, paragraph in enumerate(re.split(r"\n\s*\n", text)):
        blocks: list[str] = []

        for line in paragraph.split("\n"):
            line = line.strip()

            if not line:
                continue

            if not blocks or _BULLET_LINE.match(line):
                blocks.append(line)
            else:
                blocks[-1] = blocks[-1] + " " + line   # re-join PDF line wraps

        for b_index, block in enumerate(blocks):
            for s_index, sentence in enumerate(_split_sentences(block)):
                if s_index:
                    sep = " "
                elif b_index:
                    sep = "\n"
                elif p_index:
                    sep = "\n\n"
                else:
                    sep = " "

                pieces = (
                    _split_long_sentence(sentence, chunk_size)
                    if len(sentence) > chunk_size
                    else [sentence]
                )

                for k, piece in enumerate(pieces):
                    units.append((piece, sep if k == 0 else " "))

    return units


def _render(units: list[tuple[str, str]]) -> str:
    out = ""

    for i, (sentence, sep) in enumerate(units):
        out += sentence if i == 0 else sep + sentence

    return out


def split_into_passages(
    text: str,
    chunk_size: int = 1200,
    overlap: int = 200,
) -> list[str]:
    """
    Split text into bounded passages, preserving paragraphs where possible.

    Passages always start and end on sentence boundaries, and the overlap
    repeats WHOLE trailing sentences (never a cut-off word or fragment).
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "overlap must be non-negative and smaller than chunk_size."
        )

    text = clean_text(text)

    if not text:
        return []

    chunks: list[str] = []
    current: list[tuple[str, str]] = []
    current_len = 0

    for sentence, sep in _units(text, chunk_size):
        added = len(sentence) + (len(sep) if current else 0)

        if current and current_len + added > chunk_size:
            chunks.append(_render(current))

            tail: list[tuple[str, str]] = []
            tail_len = 0

            for unit in reversed(current):
                if tail_len + len(unit[0]) + 1 > overlap:
                    break

                tail.insert(0, unit)
                tail_len += len(unit[0]) + 1

            if tail and tail_len + len(sep) + len(sentence) > chunk_size:
                tail = []

            current = list(tail)
            current_len = len(_render(current)) if current else 0
            added = len(sentence) + (len(sep) if current else 0)

        current.append((sentence, sep))
        current_len += added

    if current:
        chunks.append(_render(current))

    # Safety net: every chunk respects the configured size.
    final_chunks = []

    for chunk in chunks:
        if len(chunk) <= chunk_size:
            final_chunks.append(chunk)
        else:
            final_chunks.extend(_split_long_sentence(chunk, chunk_size))

    return final_chunks


def chunk_pages(
    pages: list[dict],
    chunk_size: int = 1200,
    overlap: int = 200,
) -> list[dict]:
    """Create research-paper chunks with page and section metadata."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "overlap must be non-negative and smaller than chunk_size."
        )

    # 1. Validate and clean every page.
    cleaned: list[tuple[int, str]] = []

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

        if (
            not isinstance(page_number, int)
            or isinstance(page_number, bool)
            or page_number < 1
        ):
            raise ValueError("Page number must be a positive integer.")

        cleaned.append((page_number, clean_text(raw_text)))

    # 2. Remove running headers/footers, then repair sentences cut by page breaks.
    repeated = _repeated_lines([text for _, text in cleaned])
    texts = [remove_boilerplate_lines(text, repeated) for _, text in cleaned]
    texts = _move_dangling_fragments(texts)

    chunks = []

    # The section carries over to following pages (it used to reset to
    # "General" on every page, so most chunks lost their section).
    current_section = "General"

    for (page_number, _), text in zip(cleaned, texts):
        if not text:
            continue

        section_lines = []

        def save_section(lines, section_name, page_number=page_number):
            section_text = "\n".join(lines).strip()

            if not section_text:
                return

            passages = split_into_passages(
                section_text,
                chunk_size=chunk_size,
                overlap=overlap,
            )

            for passage in passages:
                chunks.append({
                    "chunk_id": len(chunks),
                    "page": page_number,
                    "section": section_name,
                    "text": passage,
                })

        for line in _merge_wrapped_headings(text.splitlines()):
            title = _heading_title(line)

            # An ALL-CAPS line before any real heading is the paper TITLE:
            # keep it as text instead of turning it into a section name.
            if (
                title
                and current_section == "General"
                and not (is_research_heading(line) or is_numbered_heading(line))
            ):
                title = None

            if title:
                save_section(section_lines, current_section)
                section_lines = []
                current_section = title
            else:
                section_lines.append(line)

        save_section(section_lines, current_section)

    return chunks