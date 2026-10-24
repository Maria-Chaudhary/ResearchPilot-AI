
import os
import re
from typing import Any

import httpx

import os
import re
import httpx

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

MAX_CONTEXT_PASSAGES = 8
MAX_PASSAGE_CHARACTERS = 2800
MAX_ANSWER_TOKENS = 700

# Map-reduce settings. These are deliberately modest for a CPU-only laptop.
SUMMARY_BATCH_SIZE = 5
SUMMARY_CHUNK_CHARACTERS = 2400
SUMMARY_BATCH_TOKENS = 650
SUMMARY_FINAL_TOKENS = 1000


SUMMARY_PATTERNS = (
    r"\bsummar(?:y|ize|ise)\b",
    r"\boverview\b",
    r"\bwhat is this paper about\b",
    r"\bwhat is the paper about\b",
    r"\bwhat does this paper discuss\b",
    r"\bmain topic\b",
    r"\bkey contributions\b",
    r"\bwhole paper\b",
    r"\bentire paper\b",
    r"\bsummarize the article\b",
    r"\bsummarise the article\b",
)

QUESTION_TYPES = {
    "goals": (
        r"\bgoals?\b", r"\bobjectives?\b", r"\baims?\b",
        r"\bpurpose\b",
    ),
    "methodology": (
        r"\bmethod(?:s|ology)?\b", r"\bapproach\b",
        r"\bexperimental setup\b", r"\bdata collection\b",
        r"\bhow was\b", r"\bhow were\b",
    ),
    "results": (
        r"\bresults?\b", r"\bfindings?\b", r"\boutcomes?\b",
        r"\bwhat did .* find\b", r"\bperformance\b",
    ),
    "limitations": (
        r"\blimitations?\b", r"\bweakness(?:es)?\b",
        r"\bshortcomings?\b", r"\bfuture work\b",
    ),
}


def _call_groq(
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
) -> str:
    """Make one Groq request without exposing secrets to the UI."""
    api_key = GROQ_API_KEY
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is missing. Configure it in your local .env file."
        )

    response = httpx.post(
        GROQ_API_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": GROQ_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.1,
            "max_tokens": max_tokens,
        },
        timeout=90.0,
    )
    response.raise_for_status()

    data = response.json()
    answer = data["choices"][0]["message"]["content"].strip()
    if not answer:
        raise ValueError("The language model returned an empty answer.")
    return answer


def _classify_question(question: str) -> str:
    q = question.lower().strip()

    if any(re.search(pattern, q) for pattern in SUMMARY_PATTERNS):
        return "summary"

    for kind, patterns in QUESTION_TYPES.items():
        if any(re.search(pattern, q) for pattern in patterns):
            return kind

    return "specific"


def _section(item: dict[str, Any]) -> str:
    section = str(item.get("section", "general")).strip().lower()
    aliases = {
        "abstract": "abstract",
        "introduction": "introduction",
        "background": "introduction",
        "related work": "related_work",
        "literature review": "related_work",
        "methods": "methodology",
        "methodology": "methodology",
        "research methodology": "methodology",
        "materials and methods": "methodology",
        "experimental setup": "methodology",
        "results": "results",
        "findings": "results",
        "results and discussion": "results",
        "discussion": "discussion",
        "conclusion": "conclusion",
        "conclusions": "conclusion",
        "limitations": "limitations",
        "future work": "limitations",
        "references": "references",
        "bibliography": "references",
    }

    if section in aliases:
        return aliases[section]

    text = str(item.get("text", ""))
    for line in text.splitlines()[:4]:
        normalized = re.sub(r"[^a-z ]", "", line.lower()).strip()
        if normalized in aliases:
            return aliases[normalized]

    return "general"


def _is_reference(item: dict[str, Any]) -> bool:
    if item.get("is_reference") is True:
        return True
    return _section(item) == "references"


def _terms(text: str) -> set[str]:
    stop_words = {
        "the", "a", "an", "is", "are", "was", "were", "of", "to",
        "in", "on", "for", "and", "or", "with", "by", "from", "as",
        "it", "this", "that", "what", "which", "how", "does", "do",
        "did", "about", "paper", "article", "explain", "describe",
        "discuss", "discussed", "please", "tell", "me", "give",
    }
    return {
        token for token in re.findall(r"[a-zA-Z0-9]+", text.lower())
        if len(token) > 1 and token not in stop_words
    }


def _relevance_bonus(
    question: str,
    item: dict[str, Any],
    question_type: str,
) -> float:
    text = str(item.get("text", "")).lower()
    section = _section(item)
    bonus = 0.0

    if question_type == "goals":
        if re.search(r"\b(goals?|aims?|objectives?|purpose)\b", text):
            bonus += 0.20
        if re.search(
            r"\b(main|primary|principal)\s+(goal|aim|objective)",
            text,
        ):
            bonus += 0.30
        if section in {"abstract", "introduction"}:
            bonus += 0.08

    elif question_type == "methodology":
        if section == "methodology":
            bonus += 0.35
        if re.search(
            r"\b(method|methodology|procedure|experiment|dataset)\b",
            text,
        ):
            bonus += 0.15

    elif question_type == "results":
        if section == "results":
            bonus += 0.35
        if re.search(
            r"\b(results?|findings?|accuracy|performance)\b",
            text,
        ):
            bonus += 0.15

    elif question_type == "limitations":
        if section == "limitations":
            bonus += 0.35
        if re.search(
            r"\b(limitation|future work|challenge|weakness)\b",
            text,
        ):
            bonus += 0.15

    terms = _terms(question)
    passage_terms = _terms(text)
    if terms:
        bonus += 0.25 * len(terms & passage_terms) / len(terms)

    return bonus


def _rank_candidates(
    question: str,
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    kind = _classify_question(question)
    ranked = []

    for original in candidates:
        item = dict(original)
        base = float(item.get("similarity", 0.0))
        bonus = _relevance_bonus(question, item, kind)
        penalty = 1.0 if _is_reference(item) else 0.0

        item["ranking_score"] = round(base + bonus - penalty, 4)
        item["relevance_bonus"] = round(bonus, 4)
        item["is_reference"] = _is_reference(item)
        ranked.append(item)

    ranked.sort(
        key=lambda item: (
            item["ranking_score"],
            float(item.get("keyword_score", 0.0)),
            float(item.get("semantic_similarity", 0.0)),
        ),
        reverse=True,
    )
    return ranked


def _select_passages(
    ranked: list[dict[str, Any]],
    max_passages: int,
) -> list[dict[str, Any]]:
    candidates = [item for item in ranked if not _is_reference(item)]
    if not candidates:
        candidates = ranked

    selected = []
    seen_ids = set()
    seen_text = set()

    for item in candidates:
        identity = item.get(
            "chunk_id",
            (item.get("page"), item.get("text")),
        )
        normalized_text = re.sub(
            r"\W+", " ", str(item.get("text", "")).lower()
        ).strip()

        if identity in seen_ids or not normalized_text or normalized_text in seen_text:
            continue

        selected.append(item)
        seen_ids.add(identity)
        seen_text.add(normalized_text)

        if len(selected) >= max_passages:
            break

    return selected


def _format_sources(
    passages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    sources = []

    for item in passages:
        sources.append({
            "page": item.get("page", "Unknown"),
            "text": item.get("text", ""),
            "section": item.get("section", "General"),
            "similarity": item.get("similarity", 0.0),
            "ranking_score": item.get(
                "ranking_score", item.get("similarity", 0.0)
            ),
            "chunk_id": item.get("chunk_id"),
        })

    return sources


def _build_context(sources: list[dict[str, Any]]) -> str:
    blocks = []
    for i, source in enumerate(sources, start=1):
        text = str(source["text"])[:MAX_PASSAGE_CHARACTERS]
        blocks.append(
            f"[SOURCE {i} | PAGE {source['page']} | "
            f"SECTION {source['section']}]\n{text}"
        )
    return "\n\n---\n\n".join(blocks)


def _whole_document_summary(
    all_chunks: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]]]:
    """
    Map-reduce over every non-reference chunk in the indexed document.
    Returns a final summary and the original source chunks used.
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
        return "No readable content was found in this document.", []

    # Keep the original chunk order to preserve the paper's narrative.
    content_chunks.sort(
        key=lambda item: (
            int(item.get("page", 0))
            if str(item.get("page", "")).isdigit()
            else 0,
            str(item.get("chunk_id", "")),
        )
    )

    map_system = """
You are analyzing part of a research paper for a document-level summary.
Summarize only the supplied text. Preserve important claims, methods,
findings, limitations, definitions, and conclusions. Do not invent details.
Write concise notes in your own words. Keep page markers such as [Page 3]
next to facts so they can be cited later. Ignore bibliography-only material.
""".strip()

    intermediate_summaries = []
    used_chunks = []

    # MAP: every non-reference chunk is assigned to a batch.
    for start in range(0, len(content_chunks), SUMMARY_BATCH_SIZE):
        batch = content_chunks[start:start + SUMMARY_BATCH_SIZE]
        blocks = []

        for chunk in batch:
            page = chunk.get("page", "Unknown")
            section = chunk.get("section", "General")
            text = str(chunk["text"])[:SUMMARY_CHUNK_CHARACTERS]
            blocks.append(
                f"[Page {page} | Section: {section}]\n{text}"
            )

        prompt = (
            "Summarize the important information in this batch. "
            "Include all distinct useful points and their page citations. "
            "Do not add outside knowledge.\n\n"
            + "\n\n---\n\n".join(blocks)
        )

        intermediate_summaries.append(
            _call_groq(map_system, prompt, SUMMARY_BATCH_TOKENS)
        )
        used_chunks.extend(batch)

    # REDUCE: combine all intermediate summaries, using sub-batches if needed.
    reduce_system = """
You are a professional research-paper summarizer.
Combine the supplied partial summaries into one coherent paper summary.
Remove repetition while retaining distinct information. Use clear headings:
Overview, Research Goal, Methodology, Main Findings, Limitations, and
Conclusion when supported by the evidence. Do not invent missing sections.
Cite claims with the original [Page N] markers supplied in the notes.
If the evidence does not establish a point, omit it or state that it is unclear.
""".strip()

    current_summaries = intermediate_summaries

    # Reduce hierarchically so very long papers do not create one huge request.
    while len(current_summaries) > 4:
        reduced = []

        for start in range(0, len(current_summaries), 4):
            group = current_summaries[start:start + 4]
            prompt = (
                "Combine these partial paper summaries into concise notes. "
                "Preserve distinct claims and original page citations. "
                "Remove repeated points; do not add facts.\n\n"
                + "\n\n---\n\n".join(group)
            )
            reduced.append(
                _call_groq(reduce_system, prompt, SUMMARY_BATCH_TOKENS)
            )

        current_summaries = reduced

    final_prompt = (
        "Create the final professional, easy-to-understand summary of the "
        "whole research paper from these notes. Use the headings Overview, "
        "Research Goal, Methodology, Main Findings, and Conclusion where "
        "supported. Include limitations if discussed. Cite claims using "
        "the original [Page N] markers. Avoid repetition and outside facts. "
        "Do not say a detail is present if the notes do not support it.\n\n"
        + "\n\n---\n\n".join(current_summaries)
    )

    final_summary = _call_groq(
        reduce_system,
        final_prompt,
        SUMMARY_FINAL_TOKENS,
    )
    return final_summary, used_chunks


def answer_question(
    store,
    question: str,
    top_k: int = 5,
) -> dict[str, Any]:
    """
    Backwards-compatible API:
    answer_question(store, question, top_k=5)

    Always returns:
      answer: str
      sources: list[dict]
      extracted_text: str
    """
    if not isinstance(question, str) or not question.strip():
      raise ValueError("Question cannot be empty")

    if (
    isinstance(top_k, bool)
    or not isinstance(top_k, int)
    or top_k < 1
):
      raise ValueError("top_k must be a positive integer")
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    question = question.strip()
    all_chunks = list(getattr(store, "chunks", []) or [])

    if not all_chunks:
        return {
            "answer": "Please upload a research paper before asking a question.",
            "sources": [],
            "extracted_text": "",
        }

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return {
            "answer": (
                "GROQ_API_KEY is missing. Configure it in your local .env "
                "file and restart the app."
            ),
            "sources": [],
            "extracted_text": "",
        }

    # Whole-paper summaries must process all document chunks, not just top-k.
    if _classify_question(question) == "summary":
        try:
            summary, used_chunks = _whole_document_summary(all_chunks)
            sources = _format_sources(used_chunks)
            extracted_text = "\n\n".join(
                f"[Page {s['page']}]\n{s['text']}" for s in sources
            )
            return {
                "answer": summary,
                "sources": sources,
                "extracted_text": extracted_text,
            }
        except (
            httpx.HTTPError, KeyError, IndexError,
            TypeError, ValueError, RuntimeError,
        ) as exc:
            print(f"Whole-paper summarization failed: {type(exc).__name__}: {exc}")
            return {
                "answer": (
                    "I couldn't complete the whole-paper summary. "
                    "Please try again; your indexed document is still available."
                ),
                "sources": [],
                "extracted_text": "",
            }

    # Focused questions use retrieval across the complete indexed collection.
    index = getattr(store, "index", None)
    index_size = getattr(index, "ntotal", len(all_chunks))
    candidate_count = min(index_size, max(top_k * 4, 20))

    candidates = store.search(
        question,
        top_k=candidate_count,
        candidate_multiplier=8,
    )
    ranked = _rank_candidates(question, candidates)
    selected = _select_passages(
        ranked,
        min(MAX_CONTEXT_PASSAGES, max(top_k, 5)),
    )
    sources = _format_sources(selected)
    extracted_text = "\n\n".join(
        f"[Page {s['page']}]\n{s['text']}" for s in sources
    )

    if not sources:
        return {
            "answer": (
                "I couldn't find relevant evidence in the extracted text. "
                "Please check the PDF extraction or rephrase your question."
            ),
            "sources": [],
            "extracted_text": "",
        }

    system_prompt = """
You are a professional research-paper assistant.
Answer in clear, natural, easy-to-understand wording.

Use only the supplied passages as factual evidence.
Cite substantive claims using [Page N], based on the source labels.
Combine evidence from multiple passages when useful.
Do not invent facts, statistics, methods, results, or conclusions.
Do not repeat the same idea in multiple sentences.
Do not reproduce long source passages verbatim.
If evidence is insufficient, say so clearly.
Do not mention similarity scores or internal retrieval mechanics.
""".strip()

    user_prompt = (
        f"QUESTION:\n{question}\n\n"
        f"EVIDENCE:\n{_build_context(sources)}\n\n"
        "Answer the question directly and professionally. Paraphrase the "
        "evidence, remove repetition, and include page citations."
    )

    try:
        answer = _call_groq(
            system_prompt,
            user_prompt,
            MAX_ANSWER_TOKENS,
        )
    except (
        httpx.HTTPError, KeyError, IndexError,
        TypeError, ValueError, RuntimeError,
    ) as exc:
        print(f"Answer generation failed: {type(exc).__name__}: {exc}")
        answer = (
            "I couldn't generate an answer right now. The supporting source "
            "passages are available below."
        )

    return {
        "answer": answer,
        "sources": sources,
        "extracted_text": extracted_text,
    }
