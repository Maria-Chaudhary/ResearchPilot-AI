
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
    "describe", "tell", "me", "give", "show",
}

REFERENCE_HEADINGS = {
    "references",
    "bibliography",
    "works cited",
    "literature cited",
    "selected references",
}


def tokenize(text: str) -> list[str]:
    """Tokenize text while preserving useful technical terms."""

    words = re.findall(
        r"[a-zA-Z0-9]+(?:[-'][a-zA-Z0-9]+)*",
        str(text).lower(),
    )

    return [
        word
        for word in words
        if len(word) > 1 and word not in STOP_WORDS
    ]


def is_reference_chunk(chunk: dict[str, Any]) -> bool:
    """Detect reference sections conservatively."""

    section = str(chunk.get("section", "")).strip().lower()

    if section in REFERENCE_HEADINGS:
        return True

    text = str(chunk.get("text", ""))

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    if not lines:
        return False

    for line in lines[:3]:
        normalized = re.sub(
            r"[^a-z ]",
            "",
            line.lower(),
        ).strip()

        if normalized in REFERENCE_HEADINGS:
            return True

    citation_count = 0

    for line in lines:
        has_year = bool(
            re.search(r"\b(?:19|20)\d{2}[a-z]?\b", line)
        )

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


class FaissVectorStore:
    """
    Hybrid document retrieval using:

    1. FAISS IndexFlatIP for semantic retrieval.
    2. BM25 for keyword retrieval.
    3. Combined scoring for final ranking.

    Embeddings are normalized, so inner product represents
    cosine similarity.
    """

    def __init__(self):
        self.index = None
        self.chunks: list[dict[str, Any]] = []

        self.vectors: np.ndarray | None = None
        self.tokenized_chunks: list[list[str]] = []

        self.document_frequencies: Counter = Counter()
        self.average_length = 0.0
        self.dimension: int | None = None

        self.reference_flags: list[bool] = []

    # -----------------------------------------------------
    # Index construction
    # -----------------------------------------------------

    def add_chunks(self, chunks: list[dict[str, Any]]) -> None:
        """Build and replace the active document index."""

        if not isinstance(chunks, list) or not chunks:
            raise ValueError(
                "chunks must be a non-empty list."
            )

        validated = []

        for i, chunk in enumerate(chunks):
            if not isinstance(chunk, dict):
                raise ValueError(
                    f"Chunk {i} must be a dictionary."
                )

            text = chunk.get("text")

            if not isinstance(text, str) or not text.strip():
                raise ValueError(
                    f"Chunk {i} has empty text."
                )

            item = dict(chunk)
            item.setdefault("chunk_id", i)
            item["text"] = text.strip()

            validated.append(item)

        # Generate document embeddings.
        vectors = np.asarray(
            embed_texts(
                [chunk["text"] for chunk in validated]
            ),
            dtype=np.float32,
        )

        if vectors.ndim != 2:
            raise ValueError(
                "Embeddings must be a two-dimensional array."
            )

        if vectors.shape[0] != len(validated):
            raise ValueError(
                "Embedding count does not match chunk count."
            )

        if vectors.shape[1] == 0:
            raise ValueError(
                "Embedding dimension cannot be zero."
            )

        if not np.isfinite(vectors).all():
            raise ValueError(
                "Embeddings contain invalid numerical values."
            )

        # Reject zero vectors instead of silently accepting them.
        norms = np.linalg.norm(vectors, axis=1)

        if np.any(norms <= 1e-12):
            raise ValueError(
                "One or more document embeddings are zero vectors."
            )

        faiss.normalize_L2(vectors)

        dimension = vectors.shape[1]

        # Build FAISS index.
        new_index = faiss.IndexFlatIP(dimension)
        new_index.add(vectors)

        # Prepare BM25 metadata.
        tokenized_chunks = [
            tokenize(chunk["text"])
            for chunk in validated
        ]

        document_frequencies = Counter()

        for tokens in tokenized_chunks:
            document_frequencies.update(set(tokens))

        lengths = [
            len(tokens)
            for tokens in tokenized_chunks
        ]

        average_length = (
            sum(lengths) / len(lengths)
            if lengths else 0.0
        )

        reference_flags = [
            is_reference_chunk(chunk)
            for chunk in validated
        ]

        # Replace the active index only after all operations succeed.
        self.index = new_index
        self.vectors = vectors
        self.chunks = validated
        self.tokenized_chunks = tokenized_chunks
        self.document_frequencies = document_frequencies
        self.average_length = average_length
        self.dimension = dimension
        self.reference_flags = reference_flags

    # -----------------------------------------------------
    # BM25
    # -----------------------------------------------------

    def _bm25_scores(
        self,
        query_tokens: list[str],
    ) -> np.ndarray:
        """Calculate normalized BM25 scores for all chunks."""

        count = len(self.chunks)

        scores = np.zeros(
            count,
            dtype=np.float32,
        )

        if not query_tokens or count == 0:
            return scores

        k1 = 1.5
        b = 0.75

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

                document_frequency = (
                    self.document_frequencies.get(term, 0)
                )

                idf = math.log(
                    1.0
                    + (
                        count - document_frequency + 0.5
                    ) / (
                        document_frequency + 0.5
                    )
                )

                denominator = term_frequency + k1 * (
                    1.0
                    - b
                    + b * document_length / average_length
                )

                score += (
                    idf
                    * term_frequency
                    * (k1 + 1.0)
                    / denominator
                )

            scores[i] = score

        maximum = float(scores.max())

        if maximum > 0:
            scores /= maximum

        return scores

    # -----------------------------------------------------
    # Search validation
    # -----------------------------------------------------

    @staticmethod
    def _validate_search_arguments(
        query: str,
        top_k: int,
        candidate_multiplier: int,
        min_similarity: float | None,
    ) -> None:

        if not isinstance(query, str) or not query.strip():
            raise ValueError(
                "query must be a non-empty string."
            )

        if (
            isinstance(top_k, bool)
            or not isinstance(top_k, int)
            or top_k < 1
        ):
            raise ValueError(
                "top_k must be a positive integer."
            )

        if (
            isinstance(candidate_multiplier, bool)
            or not isinstance(candidate_multiplier, int)
            or candidate_multiplier < 1
        ):
            raise ValueError(
                "candidate_multiplier must be a positive integer."
            )

        if min_similarity is not None:
            try:
                min_similarity = float(min_similarity)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "min_similarity must be a valid number."
                ) from exc

            if not math.isfinite(min_similarity):
                raise ValueError(
                    "min_similarity must be finite."
                )

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
        """
        Retrieve relevant passages using FAISS + BM25.

        Returns:
            List of chunks containing:
                similarity
                semantic_similarity
                keyword_score
                query_term_coverage
                is_reference
        """

        self._validate_search_arguments(
            query,
            top_k,
            candidate_multiplier,
            min_similarity,
        )

        if (
            self.index is None
            or self.vectors is None
            or not self.chunks
        ):
            return []

        query = query.strip()
        chunk_count = len(self.chunks)

        # -------------------------------------------------
        # 1. Embed query
        # -------------------------------------------------

        query_vector = np.asarray(
            embed_texts([query]),
            dtype=np.float32,
        )

        if (
            query_vector.ndim != 2
            or query_vector.shape[0] != 1
            or query_vector.shape[1] != self.dimension
        ):
            raise ValueError(
                "Query embedding dimensions do not match the index."
            )

        if not np.isfinite(query_vector).all():
            raise ValueError(
                "Query embedding contains invalid numerical values."
            )

        if np.linalg.norm(query_vector) <= 1e-12:
            raise ValueError(
                "Query embedding is a zero vector."
            )

        faiss.normalize_L2(query_vector)

        # -------------------------------------------------
        # 2. FAISS semantic retrieval
        # -------------------------------------------------

        candidate_count = min(
            chunk_count,
            max(top_k * candidate_multiplier, 20),
        )

        semantic_scores, semantic_indices = self.index.search(
            query_vector,
            candidate_count,
        )

        semantic_candidates = {}

        for score, position in zip(
            semantic_scores[0],
            semantic_indices[0],
        ):
            if position < 0:
                continue

            semantic_candidates[int(position)] = float(score)

        # -------------------------------------------------
        # 3. BM25 keyword retrieval
        # -------------------------------------------------

        query_tokens = tokenize(query)

        bm25_scores = self._bm25_scores(query_tokens)

        keyword_indices = np.argsort(
            bm25_scores
        )[::-1][:candidate_count]

        candidate_indices = set(
            semantic_candidates.keys()
        )

        for position in keyword_indices:
            position = int(position)

            if bm25_scores[position] > 0:
                candidate_indices.add(position)

        if not candidate_indices:
            return []

        # -------------------------------------------------
        # 4. Hybrid scoring
        # -------------------------------------------------

        query_terms = set(query_tokens)
        results = []

        for position in candidate_indices:
            chunk = self.chunks[position]

            semantic = semantic_candidates.get(position)

            # Calculate exact cosine similarity for BM25-only results.
            if semantic is None:
                semantic = float(
                    np.dot(
                        self.vectors[position],
                        query_vector[0],
                    )
                )

            lexical = float(bm25_scores[position])

            # Map cosine similarity [-1, 1] to [0, 1].
            semantic_normalized = (
                semantic + 1.0
            ) / 2.0

            combined = (
                0.65 * semantic_normalized
                + 0.35 * lexical
            )

            text_tokens = set(
                self.tokenized_chunks[position]
            )

            coverage = (
                len(query_terms & text_tokens) / len(query_terms)
                if query_terms else 0.0
            )

            is_reference = self.reference_flags[position]

            # Keep references available, but rank them lower.
            if is_reference:
                combined -= 0.35

            item = dict(chunk)

            item["similarity"] = round(
                float(combined), 4
            )

            item["semantic_similarity"] = round(
                float(semantic), 4
            )

            item["keyword_score"] = round(
                lexical, 4
            )

            item["query_term_coverage"] = round(
                coverage, 4
            )

            item["is_reference"] = is_reference

            results.append(item)

        # -------------------------------------------------
        # 5. Ranking
        # -------------------------------------------------

        results.sort(
            key=lambda item: (
                item["similarity"],
                item["query_term_coverage"],
                item["semantic_similarity"],
            ),
            reverse=True,
        )

        # -------------------------------------------------
        # 6. Deduplication and threshold
        # -------------------------------------------------

        unique = []
        seen = set()

        for item in results:
            normalized = re.sub(
                r"\W+",
                " ",
                item["text"].lower(),
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
