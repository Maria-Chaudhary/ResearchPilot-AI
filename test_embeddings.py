
from app.retrieval.embeddings import embed_texts

texts = [
    "Machine learning models learn patterns from data.",
    "Embeddings represent text as numerical vectors.",
    "The weather is sunny and warm today.",
]

vectors = embed_texts(texts)

print("Number of texts:", len(texts))
print("Embedding matrix shape:", vectors.shape)

similarity = float(vectors[0] @ vectors[1])
print("Similarity between first two texts:", round(similarity, 4))

print("\nEmbedding model is working!")
