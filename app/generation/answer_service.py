
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
ENV_FILE = PROJECT_ROOT / ".env"

load_dotenv(ENV_FILE)

# Keep this module-level variable so tests can mock it.
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = os.getenv(
    "GROQ_MODEL",
    "llama-3.3-70b-versatile",
)

MAX_CONTEXT_PASSAGES = 8
MAX_PASSAGE_CHARACTERS = 2800
MAX_ANSWER_TOKENS = 700

SUMMARY_BATCH_SIZE = 5
SUMMARY_CHUNK_CHARACTERS = 2400
SUMMARY_BATCH_TOKENS = 650
SUMMARY_FINAL_TOKENS = 1000


# ---------------------------------------------------------
# Groq API
# ---------------------------------------------------------

def _call_groq(
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = MAX_ANSWER_TOKENS,
) -> str:
    """Send a chat-completion request to Groq."""

    if not GROQ_API_KEY:
        raise ValueError(
            "GROQ_API_KEY is missing. Add it to your root .env file."
        )

    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.1,
        "max_tokens": max_tokens,
    }

    response = httpx.post(
        GROQ_API_URL,
        headers={
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=90.0,
    )
    response.raise_for_status()

    data = response.json()

    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(
            "Groq returned an unexpected response format."
        ) from exc

    if not isinstance(content, str) or not content.strip():
        raise ValueError("Groq returned an empty answer.")

    return content.strip()


# ---------------------------------------------------------
# General helpers
# ---------------------------------------------------------

def _page_number(chunk: dict[str, Any]) -> str:
    """Return a readable page number for a chunk."""

    page = chunk.get("page", chunk.get("page_number", "?"))

    if page is None:
        return "?"

    return str(page)


def _chunk_identity(chunk: dict[str, Any]) -> tuple:
    """Create a stable identity for deduplicating chunks."""

    chunk_id = chunk.get("chunk_id")

    if chunk_id is not None:
        return ("id", str(chunk_id))

    return (
        "content",
        _page_number(chunk),
        re.sub(r"\W+", " ", str(chunk.get("text", "")).lower()).strip(),
    )


def _section(chunk: dict[str, Any]) -> str:
    return str(chunk.get("section", "General")).strip() or "General"


def _is_reference(chunk: dict[str, Any]) -> bool:
    """Conservatively identify reference-list chunks."""

    section = _section(chunk).lower().strip()

    reference_headings = {
        "references",
        "bibliography",
        "works cited",
        "literature cited",
        "selected references",
    }

    if section in reference_headings:
        return True

    text = str(chunk.get("text", ""))
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    if not lines:
        return False

    for line in lines[:3]:
        cleaned = re.sub(r"[^a-z ]", "", line.lower()).strip()
        if cleaned in reference_headings:
            return True

    citation_count = 0

    for line in lines:
        has_year = bool(re.search(r"\b(?:19|20)\d{2}[a-z]?\b", line))
        has_citation_cue = bool(
            re.search(
                r"\b(doi:|proceedings|journal of|vol\.|pp\.|"
                r"technical report|university press)\b",
                line,
                re.IGNORECASE,
            )
        )

        if has_year and has_citation_cue:
            citation_count += 1

    return (
        len(lines) >= 4
        and citation_count >= 3
        and citation_count / len(lines) >= 0.55
    )


def _terms(text: str) -> set[str]:
    """Extract useful terms for lightweight relevance ranking."""

    stop_words = {
        "a", "an", "the", "is", "are", "was", "were", "be", "been",
        "being", "of", "to", "in", "on", "for", "and", "or", "but",
        "with", "by", "from", "at", "as", "it", "this", "that",
        "these", "those", "what", "which", "who", "when", "where",
        "why", "how", "do", "does", "did", "can", "could", "would",
        "should", "about", "paper", "article", "please", "explain",
        "describe", "tell", "me", "give", "show",
    }

    words = re.findall(
        r"[a-zA-Z0-9]+(?:[-'][a-zA-Z0-9]+)*",
        text.lower(),
    )

    return {
        word for word in words
        if len(word) > 1 and word not in stop_words
    }


def _relevance_bonus(
    question: str,
    chunk: dict[str, Any],
) -> float:
    """Add a modest exact-term relevance bonus."""

    question_terms = _terms(question)
    text = str(chunk.get("text", ""))
    text_terms = _terms(text)

    if not question_terms:
        return 0.0

    coverage = len(question_terms & text_terms) / len(question_terms)
    return 0.15 * coverage


def _rank_candidates(
    question: str,
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Rank retrieved candidates without losing their source metadata."""

    ranked = []

    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue

        text = candidate.get("text")

        if not isinstance(text, str) or not text.strip():
            continue

        item = dict(candidate)

        try:
            base_score = float(
                item.get("similarity", item.get("score", 0.0))
            )
        except (TypeError, ValueError):
            base_score = 0.0

        item["_ranking_score"] = (
            base_score + _relevance_bonus(question, item)
        )

        if _is_reference(item):
            item["_ranking_score"] -= 0.35

        ranked.append(item)

    ranked.sort(
        key=lambda item: (
            item["_ranking_score"],
            float(item.get("semantic_similarity", 0.0) or 0.0),
            float(item.get("query_term_coverage", 0.0) or 0.0),
        ),
        reverse=True,
    )

    return ranked


def _select_passages(
    question: str,
    candidates: list[dict[str, Any]],
    limit: int = MAX_CONTEXT_PASSAGES,
) -> list[dict[str, Any]]:
    """Select relevant, nonduplicate passages."""

    ranked = _rank_candidates(question, candidates)

    selected = []
    seen_ids = set()
    seen_texts = set()

    for item in ranked:
        identity = _chunk_identity(item)

        normalized_text = re.sub(
            r"\W+",
            " ",
            str(item.get("text", "")).lower(),
        ).strip()

        if not normalized_text:
            continue

        if identity in seen_ids or normalized_text in seen_texts:
            continue

        seen_ids.add(identity)
        seen_texts.add(normalized_text)

        item.pop("_ranking_score", None)
        selected.append(item)

        if len(selected) >= limit:
            break

    return selected


def _format_sources(
    chunks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Format source metadata for the existing UI."""

    sources = []
    seen = set()

    for chunk in chunks:
        text = str(chunk.get("text", "")).strip()

        if not text:
            continue

        identity = _chunk_identity(chunk)

        if identity in seen:
            continue

        seen.add(identity)

        source = {
            "page": chunk.get("page", chunk.get("page_number")),
            "section": _section(chunk),
            "text": text,
        }

        if chunk.get("chunk_id") is not None:
            source["chunk_id"] = chunk["chunk_id"]

        if chunk.get("similarity") is not None:
            source["similarity"] = chunk["similarity"]

        if chunk.get("semantic_similarity") is not None:
            source["semantic_similarity"] = (
                chunk["semantic_similarity"]
            )

        sources.append(source)

    return sources


def _build_context(
    chunks: list[dict[str, Any]],
    max_chars: int = MAX_PASSAGE_CHARACTERS,
) -> str:
    """Build a page-labelled evidence context."""

    sections = []

    for chunk in chunks:
        text = str(chunk.get("text", "")).strip()

        if not text:
            continue

        if len(text) > max_chars:
            text = text[:max_chars].rsplit(" ", 1)[0] + " ..."

        sections.append(
            f"[Page {_page_number(chunk)} | "
            f"Section: {_section(chunk)}]\n{text}"
        )

    return "\n\n---\n\n".join(sections)


# ---------------------------------------------------------
# Question classification
# ---------------------------------------------------------

def _classify_question(question: str) -> str:
    """Identify whole-paper summaries and common research questions."""

    q = question.lower().strip()

    summary_patterns = [
        r"\bsummary\b",
        r"\bsummarize\b",
        r"\bsummarise\b",
        r"\boverview\b",
        r"\bwhat is this paper about\b",
        r"\bwhat does this paper discuss\b",
        r"\bgive me the gist\b",
        r"\bsummarize the whole paper\b",
        r"\bsummarise the whole paper\b",
        r"\bwhole[- ]paper summary\b",
    ]

    if any(re.search(pattern, q) for pattern in summary_patterns):
        return "summary"

    if re.search(
        r"\b(aim|aims|objective|objectives|purpose|research question)\b",
        q,
    ):
        return "goals"

    if re.search(
        r"\b(method|methods|methodology|approach|dataset|data collection|"
        r"experimental setup|algorithm|model architecture)\b",
        q,
    ):
        return "methodology"

    if re.search(
        r"\b(result|results|finding|findings|performance|accuracy|"
        r"evaluation|outcome|outcomes)\b",
        q,
    ):
        return "results"

    if re.search(
        r"\b(limitation|limitations|weakness|weaknesses|future work|"
        r"future direction|drawback|drawbacks)\b",
        q,
    ):
        return "limitations"

    return "specific"


# ---------------------------------------------------------
# Whole-paper summary
# ---------------------------------------------------------

def _whole_document_summary(
    all_chunks: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]]]:
    """
    Summarize all available non-reference chunks using map-reduce.

    Each batch is summarized, then the partial summaries are reduced
    into a final structured answer.
    """

    content_chunks = [
        chunk for chunk in all_chunks
        if str(chunk.get("text", "")).strip()
        and not _is_reference(chunk)
    ]

    if not content_chunks:
        content_chunks = [
            chunk for chunk in all_chunks
            if str(chunk.get("text", "")).strip()
        ]

    if not content_chunks:
        return "No readable document text was found.", []

    # Sort numerically by page when possible, then by chunk ID.
    def sort_key(chunk: dict[str, Any]) -> tuple:
        page = chunk.get("page", chunk.get("page_number", 0))

        try:
            page_key = (0, int(page))
        except (TypeError, ValueError):
            page_key = (1, str(page))

        chunk_id = chunk.get("chunk_id", 0)

        try:
            chunk_key = (0, int(chunk_id))
        except (TypeError, ValueError):
            chunk_key = (1, str(chunk_id))

        return page_key + chunk_key

    content_chunks = sorted(content_chunks, key=sort_key)

    partial_summaries = []

    for start in range(0, len(content_chunks), SUMMARY_BATCH_SIZE):
        batch = content_chunks[
            start:start + SUMMARY_BATCH_SIZE
        ]

        batch_context_parts = []

        for chunk in batch:
            text = str(chunk.get("text", "")).strip()

            if len(text) > SUMMARY_CHUNK_CHARACTERS:
                text = (
                    text[:SUMMARY_CHUNK_CHARACTERS]
                    .rsplit(" ", 1)[0]
                    + " ..."
                )

            batch_context_parts.append(
                f"[Page {_page_number(chunk)} | "
                f"Section: {_section(chunk)}]\n{text}"
            )

        batch_context = "\n\n---\n\n".join(batch_context_parts)

        system_prompt = (
            "You are a careful research-paper analyst. Summarize only "
            "the evidence provided. Do not invent facts, numbers, methods, "
            "or findings. Preserve important technical details and cite "
            "supporting page numbers using [Page N]. Distinguish reported "
            "findings from interpretations. If information is absent, "
            "say it is not specified in the provided passages."
        )

        user_prompt = (
            "Summarize the following section passages into concise notes "
            "that preserve the research goal, methods, evidence, findings, "
            "limitations, and conclusions when present. Keep page citations. "
            "Do not claim this batch represents the entire paper.\n\n"
            f"{batch_context}"
        )

        partial = _call_groq(
            system_prompt,
            user_prompt,
            max_tokens=SUMMARY_BATCH_TOKENS,
        )

        partial_summaries.append(
            f"Batch covering pages "
            f"{_page_number(batch[0])}–{_page_number(batch[-1])}:\n"
            f"{partial}"
        )

    # Reduce partial summaries in groups to stay within context limits.
    while len(partial_summaries) > 4:
        reduced = []

        for start in range(0, len(partial_summaries), 4):
            group = partial_summaries[start:start + 4]

            system_prompt = (
                "You combine research notes faithfully. Preserve distinct "
                "findings, important numerical results, limitations, and "
                "page citations. Do not add unsupported information."
            )

            user_prompt = (
                "Combine these partial summaries into a concise, "
                "nonredundant set of research notes. Retain page citations "
                "and contradictions or uncertainty where relevant.\n\n"
                + "\n\n---\n\n".join(group)
            )

            reduced.append(
                _call_groq(
                    system_prompt,
                    user_prompt,
                    max_tokens=SUMMARY_BATCH_TOKENS,
                )
            )

        partial_summaries = reduced

    final_system_prompt = (
        "You are ResearchPilot AI, a precise research assistant. Create "
        "a professional, readable summary using only the supplied notes. "
        "Do not fabricate information. Retain page citations for factual "
        "claims wherever available. If a requested detail is not covered, "
        "state that it was not specified in the reviewed text."
    )

    final_user_prompt = (
        "Write a structured summary of the research paper from these "
        "notes gathered across its indexed content. Use the headings:\n"
        "1. Overview\n"
        "2. Research Problem and Objectives\n"
        "3. Methodology\n"
        "4. Main Findings\n"
        "5. Limitations\n"
        "6. Conclusion\n\n"
        "Include only headings supported by the notes. Avoid repetition. "
        "Do not invent numerical results or claim that absent information "
        "was reported. Keep page citations.\n\n"
        + "\n\n---\n\n".join(partial_summaries)
    )

    final_summary = _call_groq(
        final_system_prompt,
        final_user_prompt,
        max_tokens=SUMMARY_FINAL_TOKENS,
    )

    return final_summary, content_chunks


# ---------------------------------------------------------
# Main public function
# ---------------------------------------------------------

def answer_question(
    store: Any,
    question: str,
    top_k: int = 5,
) -> dict[str, Any]:
    """
    Answer a question using the indexed document.

    The returned keys remain compatible with the UI:
      answer: generated answer or readable error
      sources: supporting source metadata
      extracted_text: original retrieved evidence
    """

    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question cannot be empty")

    if (
        isinstance(top_k, bool)
        or not isinstance(top_k, int)
        or top_k < 1
    ):
        raise ValueError("top_k must be a positive integer")

    question = question.strip()

    empty_result = {
        "answer": "",
        "sources": [],
        "extracted_text": "",
    }

    if not GROQ_API_KEY:
      raise RuntimeError(
        "GROQ_API_KEY is missing. Add it to your root .env file."
    )

    all_chunks = list(getattr(store, "chunks", []) or [])

    all_chunks = list(getattr(store, "chunks", []) or [])

# Some test doubles expose search() without exposing chunks.
# Allow those stores to answer through their search interface.
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

        # Whole-paper summaries must process the complete indexed content,
        # not only the most similar top-k passages.
        if question_type == "summary":
            summary, used_chunks = _whole_document_summary(all_chunks)

            return {
                "answer": summary,
                "sources": _format_sources(used_chunks),
                "extracted_text": _build_context(
                    used_chunks,
                    max_chars=SUMMARY_CHUNK_CHARACTERS,
                ),
            }

        # Retrieve a broader candidate set before selecting evidence.
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

        selected = _select_passages(
            question,
            candidates,
            limit=min(max(top_k, 1), MAX_CONTEXT_PASSAGES),
        )

        if not selected:
            return {
                **empty_result,
                "answer": (
                    "I could not find sufficiently relevant passages "
                    "in the indexed document to answer this question."
                ),
            }

        context = _build_context(selected)

        system_prompt = (
            "You are ResearchPilot AI, a research-paper question-answering "
            "assistant. Answer in clear, professional language using only "
            "the supplied document evidence. Synthesize the evidence in "
            "your own words instead of copying long passages. Cite factual "
            "claims with the provided page markers, such as [Page 2]. "
            "Never invent facts, results, numbers, quotations, or page "
            "citations. If the evidence is insufficient, say so clearly. "
            "If sources disagree, describe the disagreement."
        )

        user_prompt = (
            f"Question: {question}\n\n"
            "Use the following retrieved passages as evidence. Give a "
            "direct answer first, then explain important supporting details. "
            "Include page citations for claims. If the passages do not "
            "answer the question, clearly state the limitation.\n\n"
            f"Document evidence:\n{context}"
        )

        answer = _call_groq(
            system_prompt,
            user_prompt,
            max_tokens=MAX_ANSWER_TOKENS,
        )

        return {
            "answer": answer,
            "sources": _format_sources(selected),
            "extracted_text": context,
        }

    except Exception as exc:
        # Keep the UI response structure stable and avoid exposing secrets.
        message = str(exc).strip() or "An unexpected error occurred."

        return {
            **empty_result,
            "answer": f"Unable to generate an answer: {message}",
        }
