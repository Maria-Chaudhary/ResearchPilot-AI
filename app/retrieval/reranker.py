"""
Second-stage reranker.

Stage 1 (FAISS + BM25) is fast but scores the query and passage
independently. A cross-encoder reads the query and passage TOGETHER, which
is far better at deciding "does this passage actually answer the question?".

Output scores are probabilities in [0, 1]; they are comparable across
queries, so they can also be used as an evidence gate.
"""

from __future__ import annotations

import logging
import math
import os
import threading
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_RERANKER_MODEL = os.getenv(
    "RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

# Cross-encoders truncate at ~512 tokens. Long chunks are scored in
# overlapping windows and the best window wins, so the answer-bearing tail
# of a long chunk is not silently ignored.
WINDOW_CHARS = 1200
WINDOW_OVERLAP = 200
MAX_WINDOWS = 3


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


def _windows(text: str) -> list[str]:
    text = text.strip()
    if len(text) <= WINDOW_CHARS:
        return [text]

    step = WINDOW_CHARS - WINDOW_OVERLAP
    out, start = [], 0
    while start < len(text) and len(out) < MAX_WINDOWS:
        out.append(text[start:start + WINDOW_CHARS])
        if start + WINDOW_CHARS >= len(text):
            break
        start += step
    return out


class CrossEncoderReranker:
    def __init__(
        self,
        model_name: str | None = None,
        batch_size: int = 16,
        max_length: int = 512,
        device: str | None = None,
        model: Any = None,  # injectable for tests
    ):
        self.model_name = model_name or DEFAULT_RERANKER_MODEL
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = device
        self._model = model
        self._load_failed = False
        self._returns_logits = True
        self._lock = threading.Lock()

    def _load(self) -> bool:
        if self._model is not None:
            return True
        if self._load_failed:
            return False
        with self._lock:
            if self._model is not None:
                return True
            try:
                from sentence_transformers import CrossEncoder

                self._model = CrossEncoder(
                    self.model_name,
                    max_length=self.max_length,
                    device=self.device,
                )
            except Exception as exc:  # missing package, no network, bad name
                self._load_failed = True
                logger.warning(
                    "Reranker '%s' could not be loaded (%s). "
                    "Retrieval will continue WITHOUT reranking.",
                    self.model_name, exc,
                )
                return False
        return True

    @property
    def available(self) -> bool:
        return self._load()

    def _predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        kwargs = {"batch_size": self.batch_size, "show_progress_bar": False}

        # Ask for raw logits so we control the sigmoid. The argument name
        # differs between sentence-transformers versions.
        try:
            from torch import nn

            for name in ("activation_fn", "activation_fct"):
                try:
                    raw = self._model.predict(pairs, **kwargs, **{name: nn.Identity()})
                    self._returns_logits = True
                    return [float(v) for v in raw]
                except TypeError:
                    continue
        except ImportError:
            pass

        raw = self._model.predict(pairs, **kwargs)
        self._returns_logits = False  # library default activation already applied
        return [float(v) for v in raw]

    def score(self, query: str, texts: list[str]) -> list[float]:
        """Relevance probability in [0, 1] for each text."""
        if not texts:
            return []
        if not self._load():
            raise RuntimeError("Reranker model is not available.")

        pairs, owners = [], []
        for i, text in enumerate(texts):
            for window in _windows(str(text)):
                pairs.append((query, window))
                owners.append(i)

        raw = self._predict(pairs)

        best = [float("-inf")] * len(texts)
        for owner, value in zip(owners, raw):
            best[owner] = max(best[owner], value)

        if self._returns_logits:
            return [_sigmoid(v) for v in best]
        return [min(max(v, 0.0), 1.0) for v in best]


_default: CrossEncoderReranker | None = None
_default_lock = threading.Lock()


def get_reranker() -> CrossEncoderReranker:
    global _default
    with _default_lock:
        if _default is None:
            _default = CrossEncoderReranker()
        return _default