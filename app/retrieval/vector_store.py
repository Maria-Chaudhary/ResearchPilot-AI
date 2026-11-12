import logging
import math
import re
from collections import Counter
from typing import Any

import faiss
import numpy as np

from app.retrieval.embeddings import embed_texts
from app.retrieval.reranker import get_reranker

logger = logging.getLogger(__name__)

STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been",
    "being", "of", "to", "in", "on", "for", "and", "or", "but",
    "with", "by", "from", "at", "as", "it", "this", "that",
    "these", "those", "what", "which", "who", "when", "where",
    "why", "how", "do", "does", "did", "can", "could", "would",
    "should", "about", "paper", "article", "please", "explain",
    "describe", "tell", "me", "give", "show",
}

REFERENCE_HEADINGS = {
    "references",
    "bibliography",
    "works cited",
    "literature cited",
    "selected references",
}

REFERENCE_PENALTY = 0.35
RRF_K = 60
SECTION_MATCH_RRF_BONUS = 0.02
SECTION_MATCH_SIM_BONUS = 0.10
SECTION_MATCH_FINAL_BONUS = 0.15
MAX_INTENT_CHUNKS = 8

# Question wording -> section-title fragments that usually hold the answer.
# Works for any paper because it matches fragments ("conclusion" also matches
# "Findings, Conclusion And Recommendations" or "Summary And Conclusion").
DEFINITION_PATTERN = (
    r"\b(what is|what are|define|definition|meaning of|stands for|refers? to|concept of)\b"
)

SECTION_INTENTS = (
    (r"\b(conclusions?|conclude|concluding|takeaways?|final remarks?)\b",
     ("conclusion", "recommendation", "summary")),
    (r"\b(recommend\w*)\b", ("recommendation", "conclusion")),
    (r"\b(limitations?|weakness\w*|drawbacks?|shortcomings?|threats? to validity)\b",
     ("limitation", "threat", "discussion", "future work")),
    (r"\bfuture (work|research|directions?|scope)\b",
     ("future work", "limitation", "conclusion")),
    (r"\b(objectives?|aims?|purpose|goals?)\b",
     ("objective", "introduction", "abstract", "background", "research question")),
    (r"\bresearch questions?\b", ("research question", "objective")),
    (r"\b(contributions?)\b", ("contribution", "introduction", "abstract", "conclusion")),
    (r"\b(methods?|methodology|approach|procedures?|techniques?|design|sampl\w*|"
     r"dataset|data collection|participants|instruments?|architecture|model used)\b",
     ("method", "research design", "approach", "data", "experiment", "proposed", "sampl", "model")),
    (r"\b(results?|findings?|outcomes?|accuracy|performance|evaluation|analysis)\b",
     ("result", "finding", "analysis", "discussion", "evaluation", "experiment")),
    (r"\b(abstract|summary|overview|gist)\b", ("abstract", "summary")),
    (r"\b(introduction|background|motivation|problem statement|statement of the problem|problem)\b",
     ("introduction", "background", "problem")),
    (r"\b(related work|prior work|literature|previous studies)\b",
     ("related work", "literature", "background")),
    (DEFINITION_PATTERN,
     ("introduction", "background", "literature", "concept", "definition", "abstract")),
)

END_SECTIONS = ("conclusion", "recommendation", "summary", "future work")
START_SECTIONS = ("abstract", "introduction", "background", "overview")

# Position-based intents: many papers (essays, short papers, theses) have NO
# heading called "Conclusion", the conclusion is simply the closing text.
POSITION_END_INTENT = (
    r"\b(conclusions?|conclude|concluding|takeaways?|final remarks?|recommend\w*|"
    r"future (work|research|directions?|scope))\b"
)
POSITION_START_INTENT = (
    r"\b(abstract|overview|gist|introduction|objectives?|aims?|purpose|"
    r"what is this (paper|document|study|article) about)\b"
)
# Weight of a positional match, closest-to-the-edge first. The very last chunk
# is the most likely conclusion; mid-paper chunks that merely contain the word
# "conclusions" must not outrank it.
POSITION_WEIGHTS = (2.5, 1.0, 0.4)
END_CHUNKS = START_CHUNKS = len(POSITION_WEIGHTS)

FRONT_MATTER_INTENT = (
    r"\b(title|authors?|written by|who wrote|published|publication|journal|"
    r"affiliations?|institutions?|keywords?)\b"
)


def _stem(word: str) -> str:
    """Very light suffix stripping so 'trained'/'training'/'trains' match."""
    for suffix in ("ations", "ation", "ings", "ing", "edly", "ed", "es", "s"):
        if len(word) > len(suffix) + 3 and word.endswith(suffix):
            return word[: -len(suffix)]
    return word


def tokenize(text: str) -> list[str]:
    """
    Tokenize for BM25.

    Hyphenated terms are indexed BOTH whole and split, so the query
    "self attention" matches "self-attention" and vice versa.
    """

    words = re.findall(
        r"[a-zA-Z0-9]+(?:[-'][a-zA-Z0-9]+)*",
        str(text).lower(),
    )

    tokens: list[str] = []

    for word in words:
        parts = [word]

        if "-" in word or "'" in word:
            parts += [p for p in re.split(r"[-']", word) if p]

        for part in parts:
            if len(part) > 1 and part not in STOP_WORDS:
                tokens.append(_stem(part))

    return tokens


def is_reference_chunk(chunk: dict[str, Any]) -> bool:
    """Detect reference sections conservatively."""

    section = str(chunk.get("section", "")).strip().lower()

    if section in REFERENCE_HEADINGS:
        return True

    text = str(chunk.get("text", ""))

    lines = [line.strip() for line in text.splitlines() if line.strip()]

    if not lines:
        return False

    for line in lines[:3]:
        normalized = re.sub(r"[^a-z ]", "", line.lower()).strip()

        if normalized in REFERENCE_HEADINGS:
            return True

    citation_count = 0

    for line in lines:
        has_year = bool(re.search(r"\b(?:19|20)\d{2}[a-z]?\b", line))

        has_cue = bool(
            re.search(
                r"\b(doi:|proceedings|journal of|vol\.|pp\.|"
                r"technical report|university press)\b",
                line,
                re.IGNORECASE,
            )
        )

        if has_year and has_cue:
            citation_count += 1

    return (
        len(lines) >= 4
        and citation_count >= 3
        and citation_count / len(lines) >= 0.55
    )


def _embedding_text(chunk: dict[str, Any]) -> str:
    """Prefix the section heading so 'methodology' questions can match it."""
    section = str(chunk.get("section", "") or "").strip()
    text = chunk["text"]

    if section and section.lower() not in {"unknown", "none"}:
        return f"{section}\n{text}"

    return text


def _rerank_text(chunk: dict[str, Any]) -> str:
    """Give the cross-encoder the section title too (it cannot see metadata)."""
    section = str(chunk.get("section", "") or "").strip()

    if section and section.lower() not in {"unknown", "none", "general"}:
        return f"{section}. {chunk['text']}"

    return chunk["text"]


class FaissVectorStore:
    """
    Hybrid retrieval + optional cross-encoder reranking.

    search():
        FAISS cosine + BM25, fused with Reciprocal Rank Fusion (RRF).
        RRF is used for ORDERING because it needs no score calibration.

        Each result also carries ABSOLUTE signals for gating:
            semantic_similarity  raw cosine
            query_term_coverage  fraction of query terms present
            similarity           0.7 * max(cosine, 0) + 0.3 * coverage

    rerank():
        Cross-encoder second stage. Adds reranker_score (probability).
    """

    def __init__(self, reranker=None):
        self.index = None
        self.chunks: list[dict[str, Any]] = []

        self.vectors: np.ndarray | None = None
        self.tokenized_chunks: list[list[str]] = []

        self.document_frequencies: Counter = Counter()
        self.average_length = 0.0
        self.dimension: int | None = None

        self.reference_flags: list[bool] = []
        self._reranker = reranker

    # -----------------------------------------------------
    # Index construction
    # -----------------------------------------------------

    def add_chunks(self, chunks: list[dict[str, Any]]) -> None:
        if not isinstance(chunks, list) or not chunks:
            raise ValueError("chunks must be a non-empty list.")

        validated = []

        for i, chunk in enumerate(chunks):
            if not isinstance(chunk, dict):
                raise ValueError(f"Chunk {i} must be a dictionary.")

            text = chunk.get("text")

            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"Chunk {i} has empty text.")

            item = dict(chunk)
            item.setdefault("chunk_id", i)
            item["text"] = text.strip()

            validated.append(item)

        vectors = np.asarray(
            embed_texts([_embedding_text(c) for c in validated]),
            dtype=np.float32,
        )

        if vectors.ndim != 2:
            raise ValueError("Embeddings must be a two-dimensional array.")

        if vectors.shape[0] != len(validated):
            raise ValueError("Embedding count does not match chunk count.")

        if vectors.shape[1] == 0:
            raise ValueError("Embedding dimension cannot be zero.")

        if not np.isfinite(vectors).all():
            raise ValueError("Embeddings contain invalid numerical values.")

        norms = np.linalg.norm(vectors, axis=1)

        if np.any(norms <= 1e-12):
            raise ValueError("One or more document embeddings are zero vectors.")

        faiss.normalize_L2(vectors)

        dimension = vectors.shape[1]

        new_index = faiss.IndexFlatIP(dimension)
        new_index.add(vectors)

        tokenized_chunks = [tokenize(_embedding_text(c)) for c in validated]

        document_frequencies: Counter = Counter()

        for tokens in tokenized_chunks:
            document_frequencies.update(set(tokens))

        lengths = [len(tokens) for tokens in tokenized_chunks]
        average_length = sum(lengths) / len(lengths) if lengths else 0.0

        reference_flags = [is_reference_chunk(c) for c in validated]

        self.index = new_index
        self.vectors = vectors
        self.chunks = validated
        self.tokenized_chunks = tokenized_chunks
        self.document_frequencies = document_frequencies
        self.average_length = average_length
        self.dimension = dimension
        self.reference_flags = reference_flags

        report = self.index_report()
        logger.info(
            "Indexed %d chunks; sections=%s",
            report["chunks"],
            [f'{x["section"]}({x["chunks"]})' for x in report["sections"]][:25],
        )

    def index_report(self) -> dict[str, Any]:
        """What was indexed: use it to check that headings were detected."""
        sections: dict[str, dict[str, Any]] = {}

        for chunk in self.chunks:
            name = str(chunk.get("section", "") or "Unknown")
            entry = sections.setdefault(name, {"section": name, "chunks": 0, "pages": []})
            entry["chunks"] += 1
            entry["pages"].append(chunk.get("page"))

        for entry in sections.values():
            pages = [p for p in entry["pages"] if isinstance(p, int)]
            entry["pages"] = [min(pages), max(pages)] if pages else []

        return {
            "chunks": len(self.chunks),
            "sections": list(sections.values()),
            "has_conclusion_section": any("conclusion" in n.lower() for n in sections),
        }

    # -----------------------------------------------------
    # BM25 (raw scores; callers decide how to use them)
    # -----------------------------------------------------

    def _bm25_scores(self, query_tokens: list[str]) -> np.ndarray:
        count = len(self.chunks)
        scores = np.zeros(count, dtype=np.float32)

        if not query_tokens or count == 0:
            return scores

        k1, b = 1.5, 0.75
        average_length = max(self.average_length, 1.0)
        unique_query_terms = set(query_tokens)

        for i, tokens in enumerate(self.tokenized_chunks):
            if not tokens:
                continue

            frequencies = Counter(tokens)
            document_length = len(tokens)
            score = 0.0

            for term in unique_query_terms:
                term_frequency = frequencies.get(term, 0)

                if term_frequency == 0:
                    continue

                document_frequency = self.document_frequencies.get(term, 0)

                idf = math.log(
                    1.0
                    + (count - document_frequency + 0.5)
                    / (document_frequency + 0.5)
                )

                denominator = term_frequency + k1 * (
                    1.0 - b + b * document_length / average_length
                )

                score += idf * term_frequency * (k1 + 1.0) / denominator

            scores[i] = score

        return scores

    # -----------------------------------------------------
    # Question intent -> section / front matter
    # -----------------------------------------------------

    def _intent_positions(self, query: str) -> dict[int, float]:
        """Chunks that probably hold the answer, judged from the question wording."""
        q = query.lower()
        sections = [str(c.get("section", "")).lower() for c in self.chunks]
        content = [i for i in range(len(self.chunks)) if not self.reference_flags[i]]

        wanted: set[str] = set()
        generic: set[str] = set()

        for pattern, fragments in SECTION_INTENTS:
            if re.search(pattern, q):
                # The generic "what is / define" rule must not override a
                # specific one ("what is the conclusion" is about the conclusion).
                (generic if pattern is DEFINITION_PATTERN else wanted).update(fragments)

        wanted = wanted or generic

        weights: dict[int, float] = {}

        def add(position: int, weight: float) -> None:
            weights[position] = max(weights.get(position, 0.0), weight)

        if wanted:
            for i in [
                i for i in content
                if sections[i] and any(f in sections[i] for f in wanted)
            ][:MAX_INTENT_CHUNKS]:
                add(i, 1.0)

        # Heading-less papers: fall back to WHERE the text sits.
        def has_section(fragments: tuple[str, ...]) -> bool:
            return any(any(f in sec for f in fragments) for sec in sections)

        if content:
            if re.search(POSITION_END_INTENT, q) and not has_section(END_SECTIONS):
                for i, w in zip(reversed(content[-END_CHUNKS:]), POSITION_WEIGHTS):
                    add(i, w)                               # last chunk first

            if re.search(POSITION_START_INTENT, q) and not has_section(START_SECTIONS):
                for i, w in zip(content[:START_CHUNKS], POSITION_WEIGHTS):
                    add(i, w)                               # first chunk first

        if re.search(FRONT_MATTER_INTENT, q):               # title / authors live on page 1
            def page_key(i: int):
                try:
                    return (int(self.chunks[i].get("page", 10**6)), i)
                except (TypeError, ValueError):
                    return (10**6, i)

            for i, w in zip(sorted(range(len(self.chunks)), key=page_key)[:2], (1.5, 1.0)):
                add(i, w)

        return weights

    # -----------------------------------------------------
    # Search validation
    # -----------------------------------------------------

    @staticmethod
    def _validate_search_arguments(
        query, top_k, candidate_multiplier, min_similarity
    ) -> None:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string.")

        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k must be a positive integer.")

        if (
            isinstance(candidate_multiplier, bool)
            or not isinstance(candidate_multiplier, int)
            or candidate_multiplier < 1
        ):
            raise ValueError("candidate_multiplier must be a positive integer.")

        if min_similarity is not None:
            try:
                value = float(min_similarity)
            except (TypeError, ValueError) as exc:
                raise ValueError("min_similarity must be a valid number.") from exc

            if not math.isfinite(value):
                raise ValueError("min_similarity must be finite.")

    # -----------------------------------------------------
    # Search
    # -----------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = 5,
        min_similarity: float | None = None,
        candidate_multiplier: int = 5,
    ) -> list[dict[str, Any]]:
        self._validate_search_arguments(
            query, top_k, candidate_multiplier, min_similarity
        )

        if self.index is None or self.vectors is None or not self.chunks:
            return []

        query = query.strip()
        chunk_count = len(self.chunks)

        # 1. Embed query ------------------------------------------------
        query_vector = np.asarray(embed_texts([query]), dtype=np.float32)

        if (
            query_vector.ndim != 2
            or query_vector.shape[0] != 1
            or query_vector.shape[1] != self.dimension
        ):
            raise ValueError("Query embedding dimensions do not match the index.")

        if not np.isfinite(query_vector).all():
            raise ValueError("Query embedding contains invalid numerical values.")

        if np.linalg.norm(query_vector) <= 1e-12:
            raise ValueError("Query embedding is a zero vector.")

        faiss.normalize_L2(query_vector)

        # 2. Candidate generation ---------------------------------------
        candidate_count = min(chunk_count, max(top_k * candidate_multiplier, 20))

        _, semantic_indices = self.index.search(query_vector, candidate_count)

        candidates = {int(p) for p in semantic_indices[0] if p >= 0}

        query_tokens = tokenize(query)
        bm25_raw = self._bm25_scores(query_tokens)

        for position in np.argsort(bm25_raw)[::-1][:candidate_count]:
            if bm25_raw[int(position)] > 0:
                candidates.add(int(position))

        intent_positions = self._intent_positions(query)
        candidates |= set(intent_positions)

        if not candidates:
            return []

        # 3. Absolute signals for every candidate -----------------------
        semantic = {
            p: float(np.dot(self.vectors[p], query_vector[0])) for p in candidates
        }

        max_bm25 = float(bm25_raw.max()) if bm25_raw.size else 0.0
        query_terms = set(query_tokens)

        # 4. Reciprocal Rank Fusion for ordering ------------------------
        sem_order = sorted(candidates, key=lambda p: semantic[p], reverse=True)
        sem_rank = {p: i + 1 for i, p in enumerate(sem_order)}

        lex_order = sorted(
            (p for p in candidates if bm25_raw[p] > 0),
            key=lambda p: float(bm25_raw[p]),
            reverse=True,
        )
        lex_rank = {p: i + 1 for i, p in enumerate(lex_order)}

        results = []

        for position in candidates:
            chunk = self.chunks[position]

            coverage = (
                len(query_terms & set(self.tokenized_chunks[position]))
                / len(query_terms)
                if query_terms
                else 0.0
            )

            is_reference = self.reference_flags[position]

            rrf = 1.0 / (RRF_K + sem_rank[position])
            if position in lex_rank:
                rrf += 1.0 / (RRF_K + lex_rank[position])

            similarity = 0.7 * max(semantic[position], 0.0) + 0.3 * coverage

            weight = intent_positions.get(position, 0.0)
            section_match = weight > 0

            if section_match:
                rrf += SECTION_MATCH_RRF_BONUS * weight
                similarity += SECTION_MATCH_SIM_BONUS * min(weight, 1.5)

            if is_reference:
                rrf *= 0.3
                similarity -= REFERENCE_PENALTY

            item = dict(chunk)
            item["similarity"] = round(similarity, 4)
            item["rrf_score"] = round(rrf, 6)
            item["semantic_similarity"] = round(semantic[position], 4)
            item["keyword_score"] = round(
                float(bm25_raw[position]) / max_bm25 if max_bm25 > 0 else 0.0, 4
            )
            item["query_term_coverage"] = round(coverage, 4)
            item["is_reference"] = is_reference
            item["section_match"] = section_match
            item["intent_weight"] = weight

            results.append(item)

        results.sort(
            key=lambda item: (item["rrf_score"], item["similarity"]),
            reverse=True,
        )

        # 5. Deduplicate + threshold ------------------------------------
        unique, seen = [], set()

        for item in results:
            normalized = re.sub(r"\W+", " ", item["text"].lower()).strip()

            if not normalized or normalized in seen:
                continue

            seen.add(normalized)

            if min_similarity is not None and item["similarity"] < min_similarity:
                continue

            unique.append(item)

            if len(unique) >= top_k:
                break

        return unique

    # -----------------------------------------------------
    # Cross-encoder reranking
    # -----------------------------------------------------

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Re-score candidates with a cross-encoder.

        Adds: reranker_score (0-1), final_score, reranker_used.
        If the model is unavailable, candidates are returned unchanged with
        reranker_used=False (the failure is logged, never silent).
        """

        if not candidates:
            return []

        top_k = top_k or len(candidates)
        reranker = self._reranker or get_reranker()

        try:
            scores = reranker.score(query, [_rerank_text(c) for c in candidates])
        except Exception as exc:
            logger.warning("Reranking skipped: %s", exc)
            return [dict(c, reranker_used=False) for c in candidates[:top_k]]

        reranked = []

        for candidate, score in zip(candidates, scores):
            item = dict(candidate)

            # The cross-encoder dominates; the first-stage signal breaks ties.
            final = 0.8 * score + 0.2 * float(candidate.get("similarity", 0.0))

            if candidate.get("section_match"):
                final += SECTION_MATCH_FINAL_BONUS * min(
                    float(candidate.get("intent_weight", 1.0)), 2.0
                )

            if candidate.get("is_reference"):
                final -= REFERENCE_PENALTY

            item["reranker_score"] = round(float(score), 4)
            item["final_score"] = round(final, 4)
            item["reranker_used"] = True

            reranked.append(item)

        reranked.sort(
            key=lambda i: (i["final_score"], i["reranker_score"]), reverse=True
        )

        return reranked[:top_k]