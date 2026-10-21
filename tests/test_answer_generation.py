
from unittest.mock import patch

import pytest

from app.generation.answer_service import answer_question


class FakeStore:
    def search(self, query, top_k=5):
        return [
            {
                "page": 2,
                "text": "The study evaluates machine learning models.",
                "similarity": 0.85,
            }
        ]


def test_empty_question_is_rejected():
    with pytest.raises(ValueError, match="Question cannot be empty"):
        answer_question(FakeStore(), "")


def test_invalid_top_k_is_rejected():
    with pytest.raises(ValueError, match="top_k must be a positive integer"):
        answer_question(FakeStore(), "What is the study about?", top_k=0)


def test_answer_includes_sources():
    fake_response = {
        "choices": [
            {
                "message": {
                    "content": "The study evaluates machine learning models. [Page 2]"
                }
            }
        ]
    }

    with (
        patch("app.generation.answer_service.GROQ_API_KEY", "test-key"),
        patch(
            "app.generation.answer_service.httpx.post"
        ) as mock_post,
    ):
        mock_post.return_value.raise_for_status.return_value = None
        mock_post.return_value.json.return_value = fake_response

        result = answer_question(
            FakeStore(),
            "What does the study evaluate?",
        )

    assert "machine learning models" in result["answer"]
    assert result["sources"]
    assert result["sources"][0]["page"] == 2


def test_missing_api_key_is_reported():
    with patch("app.generation.answer_service.GROQ_API_KEY", None):
        with pytest.raises(RuntimeError, match="GROQ_API_KEY is missing"):
            answer_question(FakeStore(), "What does the study evaluate?")
