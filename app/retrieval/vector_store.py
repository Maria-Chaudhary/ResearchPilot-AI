
import faiss
import numpy as np

from app.retrieval.embeddings import embed_texts


class FaissVectorStore:
    """Store research-paper embeddings and retrieve relevant passages."""

    def __init__(self):
        self.index = None
        self.chunks = []

    def add_chunks(self, chunks: list[dict]) -> None:
        """Embed valid chunks and add them to the FAISS index."""

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

        texts = [chunk["text"] for chunk in valid_chunks]

        embeddings = np.asarray(
            embed_texts(texts),
            dtype=np.float32,
        )

        if embeddings.ndim != 2:
            raise ValueError("Embeddings must be a two-dimensional array.")

        if embeddings.shape[0] != len(valid_chunks):
            raise ValueError(
                "The number of embeddings does not match the chunks."
            )

        if embeddings.shape[1] == 0:
            raise ValueError("Embedding vectors cannot be empty.")

        if not np.isfinite(embeddings).all():
            raise ValueError("Embeddings contain invalid numeric values.")

        embeddings = np.ascontiguousarray(embeddings)

        if self.index is not None:
            if embeddings.shape[1] != self.index.d:
                raise ValueError(
                    "Embedding dimensions do not match the index."
                )

        if self.index is None:
            self.index = faiss.IndexFlatIP(embeddings.shape[1])

        self.index.add(embeddings)
        self.chunks.extend(valid_chunks)

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Return the highest-scoring passages for a question."""

        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query cannot be empty.")

        if (
            not isinstance(top_k, int)
            or isinstance(top_k, bool)
            or top_k <= 0
        ):
            raise ValueError("top_k must be a positive integer.")

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
            raise ValueError(
                "The query embedding has an unexpected shape."
            )

        if not np.isfinite(query_embedding).all():
            raise ValueError("Query embedding contains invalid values.")

        query_embedding = np.ascontiguousarray(query_embedding)

        k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query_embedding, k)

        results = []

        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue

            result = self.chunks[int(idx)].copy()
            result["similarity"] = float(score)
            results.append(result)

        return results
