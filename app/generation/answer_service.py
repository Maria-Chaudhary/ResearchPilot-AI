# app/generation/answer_service.py

import os
import re
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

MAX_CONTEXT_PASSAGES = 8
MAX_PASSAGE_CHARACTERS = 2800
MAX_ANSWER_TOKENS = 700

SUMMARY_BATCH_SIZE = 5
SUMMARY_CHUNK_CHARACTERS = 2400
SUMMARY_BATCH_TOKENS = 650
SUMMARY_FINAL_TOKENS = 1000

SYSTEM_PROMPT = """
You are ResearchPilot AI, a careful research-paper assistant.

Answer the user's question using the supplied document evidence.

Rules:
1. Use the supplied evidence as the primary source.
2. Never invent facts, statistics, findings, quotations, or page numbers.
3. If the evidence does not answer the question, say so clearly.
4. Explain technical concepts in clear, professional language.
5. Distinguish the authors' claims from your own explanation.
6. Cite supporting evidence using the page labels provided, such as [Page 4].
7. Do not claim that a method or result appears in the paper unless the
   supplied evidence supports that claim.
8. Do not begin an answer with an incomplete word or sentence fragment.
9. Do not repeat the same point unnecessarily.
10. When appropriate, structure the answer with headings and bullet points.
""".strip()


# ---------------------------------------------------------
# Groq API
# ---------------------------------------------------------

def _call_groq(
    messages: list[dict[str, str]],
    max_tokens: int = MAX_ANSWER_TOKENS,
    temperature: float = 0.2,
) -> str:
    """Send a chat-completion request to Groq and return its text."""

    if not GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY is missing. Add it to your root .env file."
        )

    payload = {
        "model": GROQ_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    try:
        response = httpx.post(
            GROQ_API_URL,
            headers=headers,
            json=payload,
            timeout=90.0,
        )
        response.raise_for_status()
        data = response.json()

    except httpx.HTTPStatusError as exc:
        status_code = exc.response.status_code
        try:
            detail = exc.response.json().get("error", {}).get(
                "message", exc.response.text
            )
        except (ValueError, AttributeError):
            detail = exc.response.text

        raise RuntimeError(
            f"Groq API returned HTTP {status_code}: {detail}"
        ) from exc

    except httpx.RequestError as exc:
        raise RuntimeError(
            f"Could not connect to the Groq API: {exc}"
        ) from exc

    choices = data.get("choices", [])
    if not choices:
        raise RuntimeError("Groq returned no answer choices.")

    content = choices[0].get("message", {}).get("content", "")
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("Groq returned an empty answer.")

    return content.strip()


# ---------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------

def _clean_passage_text(
    text: str,
    max_chars: int | None = None,
    trim_partial_start: bool = True,
) -> str:
    """
    Clean extracted PDF text and avoid cutting it in the middle of a word.

    Repairs common line-break hyphenation, normalizes spaces, and prefers
    sentence boundaries when clipping long passages.
    """

    text = str(text or "")

    # Join words split by a hyphen at a PDF line break:
    # "re-
    # search" -> "research"
    text = re.sub(
        r"(?<=[A-Za-z])-\s*\n\s*(?=[a-z])",
        "",
        text,
    )

    # Normalize line endings and horizontal whitespace.
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    if not text:
        return ""

    # If the beginning looks like a fragment, skip that fragment only
    # when a sentence boundary is found near the start.
    #
    # This is a heuristic: PDF extraction may begin in the middle of a
    # sentence because chunks overlap or a page starts mid-paragraph.
    if trim_partial_start:
        first_sentence_end = re.search(
            r'[.!?]["\')\]]?\s+(?=[A-Z0-9])',
            text[:500],
        )

        if first_sentence_end:
            first_fragment = text[:first_sentence_end.start()].strip()

            # Avoid dropping a normal opening sentence. A short fragment
            # without a sentence-ending mark is more likely to be a
            # clipped passage prefix.
            if (
                first_fragment
                and not re.search(r"[.!?][\"')\]]?$", first_fragment)
                and len(first_fragment.split()) <= 12
            ):
                text = text[first_sentence_end.end():].lstrip()

    if not text:
        return ""

    if max_chars is None or len(text) <= max_chars:
        return text

    clipped = text[:max_chars].rstrip()

    # Prefer a complete sentence if one ends reasonably close to the limit.
    sentence_ends = list(
        re.finditer(r'[.!?]["\')\]]?(?=\s|$)', clipped)
    )

    if sentence_ends:
        last_end = sentence_ends[-1].end()

        if last_end >= int(max_chars * 0.5):
            return clipped[:last_end].strip()

    # Otherwise, cut at a whitespace boundary rather than mid-word.
    last_space = clipped.rfind(" ")
    if last_space > 0:
        clipped = clipped[:last_space]

    return clipped.rstrip(" ,;:-") + " …"


# ---------------------------------------------------------
# Chunk metadata helpers
# ---------------------------------------------------------

def _page_number(chunk: dict[str, Any]) -> Any:
    """Return a displayable page number."""

    page = chunk.get("page", chunk.get("page_number", "?"))

    try:
        return int(page)
    except (TypeError, ValueError):
        return page if page not in (None, "") else "?"


def _chunk_identity(chunk: dict[str, Any]) -> str:
    """Create a stable identity for a chunk."""

    chunk_id = chunk.get("chunk_id", chunk.get("id"))

    if chunk_id is not None:
        return str(chunk_id)

    return f"{_page_number(chunk)}:{str(chunk.get('text', ''))[:100]}"


def _section(chunk: dict[str, Any]) -> str:
    """Return the section name when available."""

    section = chunk.get("section", "Unknown")
    section = str(section or "Unknown").strip()

    return section or "Unknown"


def _is_reference(chunk: dict[str, Any]) -> bool:
    """Identify reference-list chunks to reduce irrelevant retrieval."""

    section = _section(chunk).lower()
    text = str(chunk.get("text", "")).strip().lower()

    reference_sections = {
        "references",
        "bibliography",
        "works cited",
        "reference",
    }

    if section in reference_sections:
        return True

    return bool(
        re.match(
            r"^(references|bibliography|works cited)\s*$",
            text[:100],
        )
    )


def _terms(text: str) -> set[str]:
    """Extract simple alphanumeric terms for relevance checks."""

    return {
        token.lower()
        for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9_-]*", text)
        if len(token) > 1
    }


# ---------------------------------------------------------
# Retrieval ranking and passage selection
# ---------------------------------------------------------

def _relevance_bonus(
    question: str,
    chunk: dict[str, Any],
) -> float:
    """Give a small boost to chunks whose text matches the question."""

    question_terms = _terms(question)
    text = str(chunk.get("text", ""))
    chunk_terms = _terms(text)

    if not question_terms or not chunk_terms:
        return 0.0

    overlap = len(question_terms & chunk_terms) / len(question_terms)

    section = _section(chunk).lower()
    question_lower = question.lower()

    section_terms = {
        "methodology": ("method", "how", "approach", "procedure"),
        "results": ("result", "finding", "performance", "accuracy"),
        "limitations": ("limitation", "weakness", "future work"),
        "introduction": ("problem", "background", "motivation"),
        "abstract": ("summary", "overview", "main idea"),
    }

    section_bonus = 0.0

    for section_name, keywords in section_terms.items():
        if section_name in section and any(
            keyword in question_lower for keyword in keywords
        ):
            section_bonus = 0.08
            break

    return min(overlap * 0.20, 0.20) + section_bonus


def _rank_candidates(
    question: str,
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Rank candidate passages and remove exact duplicate text."""

    ranked: list[tuple[float, int, dict[str, Any]]] = []
    seen_text: set[str] = set()

    for original_index, candidate in enumerate(candidates):
        if not isinstance(candidate, dict):
            continue

        text = str(candidate.get("text", "")).strip()
        if not text:
            continue

        normalized_text = re.sub(r"\s+", " ", text).lower()

        if normalized_text in seen_text:
            continue

        seen_text.add(normalized_text)

        score_value = candidate.get(
            "score",
            candidate.get(
                "similarity",
                candidate.get("combined_score", 0.0),
            ),
        )

        try:
            base_score = float(score_value or 0.0)
        except (TypeError, ValueError):
            base_score = 0.0

        bonus = _relevance_bonus(question, candidate)

        if _is_reference(candidate):
            bonus -= 0.35

        ranked.append(
            (base_score + bonus, original_index, candidate)
        )

    ranked.sort(key=lambda item: (item[0], -item[1]), reverse=True)

    return [item[2] for item in ranked]


def _select_passages(
    question: str,
    candidates: list[dict[str, Any]],
    limit: int = MAX_CONTEXT_PASSAGES,
) -> list[dict[str, Any]]:
    """Select relevant passages while avoiding near-identical text."""

    if not candidates or limit <= 0:
        return []

    ranked = _rank_candidates(question, candidates)
    selected: list[dict[str, Any]] = []
    selected_term_sets: list[set[str]] = []
    selected_ids: set[str] = set()

    for candidate in ranked:
        identity = _chunk_identity(candidate)

        if identity in selected_ids:
            continue

        text = str(candidate.get("text", "")).strip()
        current_terms = _terms(text)

        # Skip near-duplicate passages with very high term overlap.
        duplicate = False

        for previous_terms in selected_term_sets:
            if not current_terms or not previous_terms:
                continue

            overlap = len(current_terms & previous_terms)
            denominator = min(len(current_terms), len(previous_terms))
            similarity = overlap / denominator if denominator else 0.0

            if similarity >= 0.90:
                duplicate = True
                break

        if duplicate:
            continue

        selected.append(candidate)
        selected_term_sets.append(current_terms)
        selected_ids.add(identity)

        if len(selected) >= limit:
            break

    return selected


# ---------------------------------------------------------
# Evidence formatting
# ---------------------------------------------------------

def _format_sources(
    chunks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Format passages for display in the Gradio interface."""

    sources: list[dict[str, Any]] = []

    for chunk in chunks:
        text = _clean_passage_text(
            str(chunk.get("text", "")),
            max_chars=MAX_PASSAGE_CHARACTERS,
            trim_partial_start=True,
        )

        if not text:
            continue

        sources.append(
            {
                "page": _page_number(chunk),
                "section": _section(chunk),
                "text": text,
                "chunk_id": chunk.get("chunk_id", chunk.get("id")),
            }
        )

    return sources


def _build_context(
    chunks: list[dict[str, Any]],
    max_chars: int = MAX_PASSAGE_CHARACTERS,
) -> str:
    """Build readable, page-labelled evidence for the language model."""

    sections: list[str] = []

    for chunk in chunks:
        raw_text = str(chunk.get("text", "")).strip()

        if not raw_text:
            continue

        text = _clean_passage_text(
            raw_text,
            max_chars=max_chars,
            trim_partial_start=True,
        )

        if not text:
            continue

        sections.append(
            f"[Page {_page_number(chunk)} | "
            f"Section: {_section(chunk)}]\n{text}"
        )

    return "\n\n---\n\n".join(sections)


# ---------------------------------------------------------
# Question classification
# ---------------------------------------------------------

def _classify_question(question: str) -> str:
    """Classify common research-paper question types."""

    q = question.lower().strip()

    summary_terms = (
        "summarize",
        "summarise",
        "summary",
        "overview",
        "whole paper",
        "entire paper",
        "paper about",
        "main idea",
        "overall",
    )

    goal_terms = (
        "objective",
        "objectives",
        "aim",
        "purpose",
        "research question",
        "problem statement",
        "goal of",
    )

    methodology_terms = (
        "methodology",
        "method",
        "methods",
        "approach",
        "how did",
        "how was",
        "dataset",
        "data collection",
        "model used",
        "algorithm used",
    )

    result_terms = (
        "result",
        "results",
        "finding",
        "findings",
        "accuracy",
        "performance",
        "outcome",
        "conclusion",
    )

    limitation_terms = (
        "limitation",
        "limitations",
        "weakness",
        "future work",
        "future research",
        "drawback",
        "challenge",
    )

    if any(term in q for term in summary_terms):
        return "summary"

    if any(term in q for term in goal_terms):
        return "goals"

    if any(term in q for term in methodology_terms):
        return "methodology"

    if any(term in q for term in limitation_terms):
        return "limitations"

    if any(term in q for term in result_terms):
        return "results"

    return "specific"


# ---------------------------------------------------------
# Whole-document summary
# ---------------------------------------------------------

def _whole_document_summary(
    chunks: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]]]:
    """
    Summarize all indexed chunks using a map-reduce approach.

    This deliberately uses the entire document rather than only the
    highest-ranked retrieval results.
    """

    content_chunks = [
        chunk
        for chunk in chunks
        if isinstance(chunk, dict)
        and str(chunk.get("text", "")).strip()
        and not _is_reference(chunk)
    ]

    if not content_chunks:
        content_chunks = [
            chunk
            for chunk in chunks
            if isinstance(chunk, dict)
            and str(chunk.get("text", "")).strip()
        ]

    if not content_chunks:
        return (
            "No readable document content was found to summarize.",
            [],
        )

    # Keep the document's original reading order.
    content_chunks.sort(
        key=lambda chunk: (
            _page_number(chunk)
            if isinstance(_page_number(chunk), int)
            else 0,
            _chunk_identity(chunk),
        )
    )

    batch_summaries: list[str] = []

    for start in range(0, len(content_chunks), SUMMARY_BATCH_SIZE):
        batch = content_chunks[start:start + SUMMARY_BATCH_SIZE]
        batch_texts: list[str] = []

        for chunk in batch:
            text = _clean_passage_text(
                str(chunk.get("text", "")),
                max_chars=SUMMARY_CHUNK_CHARACTERS,
                trim_partial_start=False,
            )

            if not text:
                continue

            batch_texts.append(
                f"[Page {_page_number(chunk)} | "
                f"Section: {_section(chunk)}]\n{text}"
            )

        if not batch_texts:
            continue

        batch_context = "\n\n---\n\n".join(batch_texts)

        prompt = f"""
Summarize the following research-paper passages.

Preserve the paper's actual claims and details. Include the research
problem, objectives, methods, data, important findings, and limitations
when present. Do not invent missing information. Keep relevant page
references in the notes.

PASSAGES:
{batch_context}
""".strip()

        summary = _call_groq(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            max_tokens=SUMMARY_BATCH_TOKENS,
            temperature=0.2,
        )

        batch_summaries.append(summary)

    if not batch_summaries:
        return "The document did not contain enough readable text.", content_chunks

    # Reduce the summaries in groups to avoid sending the entire document
    # to the model in a single request.
    current_summaries = batch_summaries

    while len(current_summaries) > 4:
        reduced: list[str] = []

        for start in range(0, len(current_summaries), 4):
            group = current_summaries[start:start + 4]
            prompt = """
Combine the following partial research-paper summaries into one
accurate, concise summary. Preserve key methods, findings, limitations,
and page references. Remove repetition. Do not add unsupported claims.

PARTIAL SUMMARIES:
""" + "\n\n---\n\n".join(group)

            reduced.append(
                _call_groq(
                    [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=SUMMARY_BATCH_TOKENS,
                    temperature=0.2,
                )
            )

        current_summaries = reduced

    final_prompt = f"""
Create a professional, readable summary of this research paper from
the partial summaries below.

Use these headings:
1. Overview
2. Research Problem and Objectives
3. Methodology
4. Main Findings
5. Limitations and Future Work
6. Conclusion

Include only details supported by the partial summaries. If a heading's
information is not available, say that the paper does not provide enough
information in the summarized material. Preserve page references when
available. Do not start with a sentence fragment.

PARTIAL SUMMARIES:
{"".join(chr(10) + chr(10) + "---" + chr(10) + chr(10) + item for item in current_summaries)}
""".strip()

    final_summary = _call_groq(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": final_prompt},
        ],
        max_tokens=SUMMARY_FINAL_TOKENS,
        temperature=0.2,
    )

    return final_summary, content_chunks


# ---------------------------------------------------------
# Main answer function
# ---------------------------------------------------------

def answer_question(
    question: str,
    store: Any,
    top_k: int = 5,
) -> dict[str, Any]:
    """
    Answer a question using the indexed document.

    Returns the UI-compatible keys:
    - answer
    - sources
    - extracted_text
    """

    question = str(question or "").strip()

    empty_result = {
        "answer": "",
        "sources": [],
        "extracted_text": "",
    }

    if not question:
        return {
            **empty_result,
            "answer": "Please enter a question about the indexed document.",
        }

    if not GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY is missing. Add it to your root .env file."
        )

    try:
        top_k = int(top_k)
    except (TypeError, ValueError):
        top_k = 5

    top_k = max(1, min(top_k, MAX_CONTEXT_PASSAGES))

    all_chunks = list(getattr(store, "chunks", []) or [])

    # Allow lightweight test doubles or alternative stores that expose
    # search() but do not expose their chunks directly.
    if not all_chunks and not callable(getattr(store, "search", None)):
        return {
            **empty_result,
            "answer": (
                "No document is indexed yet. Please upload a readable "
                "PDF before asking a question."
            ),
        }

    try:
        question_type = _classify_question(question)

        # Summaries use every indexed content chunk, not just top-k results.
        if question_type == "summary" and all_chunks:
            answer, summary_chunks = _whole_document_summary(all_chunks)

            return {
                "answer": answer,
                "sources": _format_sources(summary_chunks),
                "extracted_text": _build_context(summary_chunks),
            }

        # Focused questions use retrieval over the index.
        index_size = len(all_chunks)
        candidate_count = (
            min(index_size, max(top_k * 4, 20))
            if index_size
            else max(top_k * 4, 20)
        )

        candidates = store.search(
            question,
            top_k=candidate_count,
        )

        if not candidates:
            return {
                **empty_result,
                "answer": (
                    "I could not find relevant evidence in the indexed "
                    "document for this question. Try rephrasing it or "
                    "checking that the correct PDF was uploaded."
                ),
            }

        selected = _select_passages(
            question,
            candidates,
            limit=min(top_k, MAX_CONTEXT_PASSAGES),
        )

        if not selected:
            return {
                **empty_result,
                "answer": (
                    "I could not identify readable evidence relevant "
                    "to your question in the indexed document."
                ),
            }

        context = _build_context(selected)

        if not context.strip():
            return {
                **empty_result,
                "answer": (
                    "Relevant passages were retrieved, but their text "
                    "could not be prepared for answering."
                ),
            }

        prompt = f"""
QUESTION:
{question}

QUESTION TYPE:
{question_type}

DOCUMENT EVIDENCE:
{context}

Write a direct, useful answer to the question.

Requirements:
- Base factual claims on the supplied evidence.
- Cite pages in the format [Page N].
- Explain the reasoning clearly without repeating the same point.
- If the evidence is incomplete, state what cannot be confirmed.
- Do not invent details or citations.
- Begin with a complete sentence, never a clipped word or fragment.
- For a methodology question, explain the approach and steps when supported.
- For a results question, report the actual findings and metrics when present.
- For objectives, state the research aim and problem when supported.
""".strip()

        answer = _call_groq(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            max_tokens=MAX_ANSWER_TOKENS,
            temperature=0.2,
        )

        # Clean accidental whitespace or a clipped opening fragment.
        answer = answer.strip()
        answer = re.sub(r"^[\s\-–—•]+", "", answer)

        sources = _format_sources(selected)

        return {
            "answer": answer,
            "sources": sources,
            "extracted_text": context,
        }

    except Exception as exc:
        return {
            **empty_result,
            "answer": f"Unable to generate an answer: {exc}",
        }