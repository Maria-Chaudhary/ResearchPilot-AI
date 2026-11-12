"""
Test setup. Real faiss / httpx / dotenv are used when installed; otherwise
light stand-ins are used. The embedding model is ALWAYS replaced by a fast
deterministic fake so tests never download or load a model.
"""
import hashlib
import re
import sys
import types
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _install_stubs() -> None:
    try:
        import faiss  # noqa: F401
    except ImportError:
        faiss = types.ModuleType("faiss")

        def normalize_L2(v):
            v[:] = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)

        class IndexFlatIP:
            def __init__(self, d):
                self.v = np.zeros((0, d), np.float32)

            def add(self, x):
                self.v = np.vstack([self.v, x])

            def search(self, q, k):
                scores = self.v @ q[0]
                idx = np.argsort(-scores)[:k]
                return scores[idx][None, :], idx[None, :]

        faiss.normalize_L2, faiss.IndexFlatIP = normalize_L2, IndexFlatIP
        sys.modules["faiss"] = faiss

    try:
        import httpx  # noqa: F401
    except ImportError:
        httpx = types.ModuleType("httpx")
        httpx.HTTPStatusError = type("HTTPStatusError", (Exception,), {})
        httpx.RequestError = type("RequestError", (Exception,), {})
        sys.modules["httpx"] = httpx

    try:
        import dotenv  # noqa: F401
    except ImportError:
        dotenv = types.ModuleType("dotenv")
        dotenv.load_dotenv = lambda *a, **k: None
        sys.modules["dotenv"] = dotenv

    def embed_texts(texts, dim=512):
        out = np.zeros((len(texts), dim), np.float32)
        for i, text in enumerate(texts):
            for word in re.findall(r"[a-z0-9]+", text.lower()):
                out[i, int(hashlib.md5(word.encode()).hexdigest(), 16) % dim] += 1
        return out

    fake = types.ModuleType("app.retrieval.embeddings")
    fake.embed_texts = embed_texts
    sys.modules["app.retrieval.embeddings"] = fake


_install_stubs()


@pytest.fixture
def ag(monkeypatch):
    """answer_service with a fake API key and no pacing delay."""
    from app.generation import answer_service

    monkeypatch.setattr(answer_service, "GROQ_API_KEY", "test-key")
    monkeypatch.setattr(answer_service, "SUMMARY_CALL_DELAY", 0)
    return answer_service


def make_store(chunks):
    from app.retrieval.vector_store import FaissVectorStore
    from app.retrieval.reranker import CrossEncoderReranker

    class _Missing:                      # reranker "not installed": must degrade gracefully
        def score(self, *a, **k):
            raise RuntimeError("no reranker in tests")

    store = FaissVectorStore(reranker=_Missing())
    store.add_chunks(chunks)
    return store


@pytest.fixture
def structured_paper():
    """A conventional paper WITH a Conclusion heading."""
    from app.ingestion.chunking import chunk_pages

    pages = [
        {"page": 1, "text": "THE IMPACT OF ARTIFICIAL INTELLIGENCE ON PRODUCTIVITY\nMaria Khan and John Doe\nABSTRACT\nThis study examined artificial intelligence and its benefits to mankind in Nigeria."},
        {"page": 2, "text": "1. INTRODUCTION\nMcCarthy defined artificial intelligence as the science and engineering of making intelligent machines. Artificial intelligence is a branch of computer science concerned with building intelligent systems that learn and reason."},
        {"page": 4, "text": "3. RESEARCH DESIGN\nThe study employed the survey research method. Questionnaires were distributed to 400 respondents in Kano State and analysed with chi-square."},
        {"page": 10, "text": "4. FINDINGS\nFindings revealed that artificial intelligence benefits the human race and increased productivity and output in many sectors of the economy.\n5. CONCLUSION\nIt is concluded that artificial intelligence is beneficial to mankind and should be adopted widely. Artificial intelligence has simplified work in many fields.\nRECOMMENDATIONS\nGovernment should invest in artificial intelligence education and infrastructure to improve national productivity."},
        {"page": 12, "text": "REFERENCES\nAsika, N. 2009. Research Methods. Journal of Things. vol. 3 pp. 1-9.\nFreitas, A. 1999. Proceedings of AI. pp. 4-8.\nMcCarthy, J. 1955. Technical report. Journal of AI vol. 1."},
    ]
    return chunk_pages(pages, chunk_size=400, overlap=80)


@pytest.fixture
def essay_paper():
    """An essay-style paper with NO conclusion heading (the user's failing case)."""
    def c(i, page, section, text):
        return {"chunk_id": i, "page": page, "section": section, "text": text}

    return [
        c(0, 1, "Introduction", "What is artificial intelligence anyway? This essay explores the goals of the field and the problems researchers face."),
        c(1, 3, "Learning", "Programs must learn from experience. Learning requires memory organisation and the ability to index prior cases."),
        c(2, 6, "Creativity", "Creativity: scientists would agree that computers may surpass humans. Relating AI to allied fields may bring new scientific conclusions. Tale-Spin created stories."),
        c(3, 6, "Generalization", "A program that can generalise from experience would be significant. This program must draw conclusions from disparate data and connect unrelated experiences."),
        c(4, 7, "Generalization", "In summary, the central goal of AI is to build programs that learn from experience, understand new events and generalise from prior cases. Progress will depend on explaining how people organise and retrieve memories."),
        c(5, 8, "References", "Meehan, J. 1976. Tale-Spin. Yale University Press.\nSchank, R. 1982. Dynamic Memory. Cambridge University Press 1982.\nFreitas, A. 1999. Proceedings of AI.\nAsika, N. 2009. Journal of Things."),
    ]