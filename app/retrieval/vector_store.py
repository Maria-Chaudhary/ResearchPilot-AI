
import re
import math
from collections import Counter
from typing import Any

import faiss
import numpy as np

from app.retrieval.embeddings import embed_texts


STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been",
    "being", "of", "to", "in", "on", "for", "and", "or", "but",
    "with", "by", "from", "at", "as", "it", "this", "that",
    "these", "those", "what", "which", "who", "when", "where",
    "why", "how", "do", "does", "did", "can", "could", "would",
    "should", "about", "paper", "article", "please", "explain",
    "describe", "tell", "me", "give", "show"
}

REFERENCE_HEADINGS = {
    "references", "bibliography", "works cited",
    "literature cited", "selected references"
}


def tokenize(text: str) -> list[str]:
    """Tokenize text while preserving useful technical terms."""
    words = re.findall(r"[a-zA-Z0-9]+(?:[-'][a-zA-Z0-9]+)*", text.lower())
    return [
        word for word in words
        if len(word) > 1 and word not in STOP_WORDS
    ]


def is_reference_chunk(chunk: dict[str, Any]) -> bool:
    """Detect reference sections conservatively."""
    section = str(chunk.get("section", "")).strip().lower()
    if section in REFERENCE_HEADINGS:
        return True

    text = str(chunk.get("text", ""))
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return False

    if any(
        re.sub(r"[^a-z ]", "", line.lower()).strip() in REFERENCE_HEADINGS
        for line in lines[:3]
    ):
        return True

    # Only classify as bibliography when several lines look like citations.
    citation_count = 0
    for line in lines:
        has_year = bool(re.search(r"\b(?:19|20)\d{2}[a-z]?\b", line))
        has_cue = bool(re.search(
            r"\b(doi:|proceedings|journal of|vol\.|pp\.|"
            r"technical report|university press)\b",
            line,
            re.IGNORECASE,
        ))
        if has_year and has_cue:
            citation_count += 1

    return (
        len(lines) >= 4
        and citation_count >= 3
        and citation_count / len(lines) >= 0.55
    )


class FaissVectorStore:
    def __init__(self):
        self.index = None
        self.chunks: list[dict[str, Any]] = []
        self.vectors: np.ndarray | None = None
        self.tokenized_chunks: list[list[str]] = []
        self.document_frequencies: Counter = Counter()
        self.average_length = 0.0
        self.dimension = None

    def add_chunks(self, chunks: list[dict[str, Any]]) -> None:
        """Replace the current index with the complete uploaded document."""
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
            embed_texts([chunk["text"] for chunk in validated]),
            dtype=np.float32,
        )

        if vectors.ndim != 2 or vectors.shape[0] != len(validated):
            raise ValueError("Unexpected embedding dimensions.")

        faiss.normalize_L2(vectors)

        self.dimension = vectors.shape[1]
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(vectors)

        # Keep vectors so we can score every chunk, not only FAISS top-k.
        self.vectors = vectors
        self.chunks = validated

        self.tokenized_chunks = [
            tokenize(chunk["text"]) for chunk in validated
        ]

        self.document_frequencies = Counter()
        for tokens in self.tokenized_chunks:
            self.document_frequencies.update(set(tokens))

        lengths = [len(tokens) for tokens in self.tokenized_chunks]
        self.average_length = (
            sum(lengths) / len(lengths) if lengths else 0.0
        )

    def _bm25_scores(self, query_tokens: list[str]) -> np.ndarray:
        """Calculate BM25 scores over every chunk."""
        count = len(self.chunks)
        scores = np.zeros(count, dtype=np.float32)

        if not query_tokens or count == 0:
            return scores

        k1 = 1.5
        b = 0.75
        avg_len = max(self.average_length, 1.0)
        unique_query_terms = set(query_tokens)

        for i, tokens in enumerate(self.tokenized_chunks):
            if not tokens:
                continue

            frequencies = Counter(tokens)
            doc_len = len(tokens)
            score = 0.0

            for term in unique_query_terms:
                tf = frequencies.get(term, 0)
                if tf == 0:
                    continue

                df = self.document_frequencies.get(term, 0)
                idf = math.log(
                    1.0 + (count - df + 0.5) / (df + 0.5)
                )
                denominator = tf + k1 * (
                    1.0 - b + b * doc_len / avg_len
                )
                score += idf * (tf * (k1 + 1.0)) / denominator

            scores[i] = score

        # Normalize scores to 0..1 for combining with semantic similarity.
        maximum = float(scores.max()) if len(scores) else 0.0
        if maximum > 0:
            scores /= maximum

        return scores

    def search(
        self,
        query: str,
        top_k: int = 5,
        min_similarity: float | None = None,
        candidate_multiplier: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Search all chunks using semantic similarity + BM25.
        candidate_multiplier is retained for API compatibility.
        """
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string.")
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        if self.index is None or self.vectors is None or not self.chunks:
            return []

        query_vector = np.asarray(
            embed_texts([query.strip()]), dtype=np.float32
        )
        faiss.normalize_L2(query_vector)

        # Matrix multiplication evaluates every chunk in one operation.
        semantic_scores = np.dot(
            self.vectors, query_vector[0]
        ).astype(np.float32)

        query_tokens = tokenize(query)
        bm25_scores = self._bm25_scores(query_tokens)

        results = []
        query_terms = set(query_tokens)

        for i, chunk in enumerate(self.chunks):
            semantic = float(semantic_scores[i])
            lexical = float(bm25_scores[i])

            # Cosine similarity is mapped from [-1, 1] to [0, 1].
            semantic_normalized = (semantic + 1.0) / 2.0

            # Combine meaning and exact-term evidence.
            combined = 0.65 * semantic_normalized + 0.35 * lexical

            text_tokens = set(self.tokenized_chunks[i])
            overlap = (
                len(query_terms & text_tokens) / len(query_terms)
                if query_terms else 0.0
            )

            if is_reference_chunk(chunk):
                combined -= 0.35

            item = dict(chunk)
            item["similarity"] = round(combined, 4)
            item["semantic_similarity"] = round(semantic, 4)
            item["keyword_score"] = round(lexical, 4)
            item["query_term_coverage"] = round(overlap, 4)
            item["is_reference"] = is_reference_chunk(chunk)
            results.append(item)

        results.sort(
            key=lambda item: (
                item["similarity"],
                item["query_term_coverage"],
                item["semantic_similarity"],
            ),
            reverse=True,
        )

        # Avoid returning exact duplicate passages.
        unique = []
        seen = set()

        for item in results:
            normalized = re.sub(
                r"\W+", " ", item["text"].lower()
            ).strip()

            if not normalized or normalized in seen:
                continue
            seen.add(normalized)

            if (
                min_similarity is not None
                and item["similarity"] < min_similarity
            ):
                continue

            unique.append(item)
            if len(unique) >= top_k:
                break

        return unique
