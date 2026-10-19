
import faiss
import numpy as np

from app.retrieval.embeddings import embed_texts


class FaissVectorStore:
    """Store text chunk embeddings and retrieve relevant passages."""

    def __init__(self):
        self.index = None
        self.chunks = []

    def add_chunks(self, chunks: list[dict]) -> None:
        """Embed chunks and add their vectors to the FAISS index."""

        valid_chunks = [
            chunk for chunk in chunks
            if chunk.get("text", "").strip()
        ]

        if not valid_chunks:
            raise ValueError("No valid text chunks were provided.")

        texts = [chunk["text"] for chunk in valid_chunks]

        embeddings = np.asarray(
            embed_texts(texts),
            dtype=np.float32,
        )
        embeddings = np.ascontiguousarray(embeddings)

        if self.index is None:
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dimension)
        elif embeddings.shape[1] != self.index.d:
            raise ValueError("Embedding dimensions do not match the index.")

        self.index.add(embeddings)
        self.chunks.extend(valid_chunks)

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Return the most relevant chunks for a user's question."""

        if not query.strip():
            raise ValueError("Query cannot be empty.")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        if self.index is None or self.index.ntotal == 0:
            raise ValueError("No chunks have been indexed yet.")

        query_vector = np.asarray(
            embed_texts([query]),
            dtype=np.float32,
        )
        query_vector = np.ascontiguousarray(query_vector)

        k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query_vector, k)

        results = []

        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue

            result = self.chunks[int(idx)].copy()
            result["similarity"] = float(score)
            results.append(result)

        return results
