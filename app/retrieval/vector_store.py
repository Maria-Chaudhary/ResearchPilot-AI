
import faiss
import numpy as np

from app.retrieval.embeddings import embed_texts


class FaissVectorStore:
    """FAISS vector store for semantic retrieval of research passages."""

    def __init__(self):
        self.index = None
        self.chunks = []

    def add_chunks(self, chunks: list[dict]) -> None:
        """Embed and index valid text chunks."""
        if not isinstance(chunks, list):
            raise ValueError("chunks must be a list.")

        valid_chunks = []

        for chunk in chunks:
            if not isinstance(chunk, dict):
                continue

            text = chunk.get("text")

            if isinstance(text, str) and text.strip():
                valid_chunks.append(chunk.copy())

        if not valid_chunks:
            raise ValueError("No valid text chunks were provided.")

        embeddings = np.asarray(
            embed_texts([chunk["text"] for chunk in valid_chunks]),
            dtype=np.float32,
        )

        if embeddings.ndim != 2:
            raise ValueError("Embeddings must be a two-dimensional array.")

        if embeddings.shape[0] != len(valid_chunks):
            raise ValueError("Embedding count does not match chunk count.")

        if embeddings.shape[1] == 0:
            raise ValueError("Embedding vectors cannot be empty.")

        if not np.isfinite(embeddings).all():
            raise ValueError("Embeddings contain invalid numeric values.")

        embeddings = np.ascontiguousarray(embeddings)

        if self.index is not None and embeddings.shape[1] != self.index.d:
            raise ValueError("Embedding dimensions do not match the index.")

        if self.index is None:
            self.index = faiss.IndexFlatIP(embeddings.shape[1])

        self.index.add(embeddings)
        self.chunks.extend(valid_chunks)

    def search(
        self,
        query: str,
        top_k: int = 5,
        min_similarity: float | None = None,
        candidate_multiplier: int = 3,
    ) -> list[dict]:
        """Retrieve ranked passages using cosine similarity.

        Embeddings are normalized, so inner-product scores correspond to
        cosine similarity. Set min_similarity only after evaluating your
        own dataset; overly high thresholds can remove useful evidence.
        """
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query cannot be empty.")

        if (
            not isinstance(top_k, int)
            or isinstance(top_k, bool)
            or top_k <= 0
        ):
            raise ValueError("top_k must be a positive integer.")

        if (
            not isinstance(candidate_multiplier, int)
            or isinstance(candidate_multiplier, bool)
            or candidate_multiplier < 1
        ):
            raise ValueError("candidate_multiplier must be at least 1.")

        if min_similarity is not None:
            if not isinstance(min_similarity, (int, float)):
                raise ValueError("min_similarity must be a number or None.")
            if not -1.0 <= min_similarity <= 1.0:
                raise ValueError("min_similarity must be between -1 and 1.")

        if self.index is None or self.index.ntotal == 0:
            raise ValueError("No chunks have been indexed yet.")

        query_embedding = np.asarray(
            embed_texts([query.strip()]),
            dtype=np.float32,
        )

        if (
            query_embedding.ndim != 2
            or query_embedding.shape[0] != 1
            or query_embedding.shape[1] != self.index.d
        ):
            raise ValueError("Query embedding has an unexpected shape.")

        if not np.isfinite(query_embedding).all():
            raise ValueError("Query embedding contains invalid values.")

        query_embedding = np.ascontiguousarray(query_embedding)

        candidate_count = min(
            self.index.ntotal,
            max(top_k, top_k * candidate_multiplier),
        )

        scores, indices = self.index.search(
            query_embedding,
            candidate_count,
        )

        results = []
        seen_text = set()

        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue

            similarity = float(score)

            if (
                min_similarity is not None
                and similarity < min_similarity
            ):
                continue

            chunk = self.chunks[int(idx)].copy()
            normalized_text = " ".join(
                chunk.get("text", "").lower().split()
            )

            if normalized_text in seen_text:
                continue

            seen_text.add(normalized_text)
            chunk["similarity"] = similarity
            results.append(chunk)

            if len(results) >= top_k:
                break

        return results
