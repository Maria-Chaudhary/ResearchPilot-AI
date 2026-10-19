
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Load the model once, using your CPU.
_model = None


def get_model() -> SentenceTransformer:
    """Load the embedding model only when first needed."""
    global _model

    if _model is None:
        _model = SentenceTransformer(MODEL_NAME, device="cpu")

    return _model


def embed_texts(texts: list[str]):
    """Convert text passages into normalized numerical vectors."""
    if not texts:
        raise ValueError("texts must contain at least one item.")

    if any(not text.strip() for text in texts):
        raise ValueError("texts must not contain empty strings.")

    model = get_model()

    return model.encode(
        texts,
        batch_size=8,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
