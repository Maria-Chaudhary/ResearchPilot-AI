
import re


def evaluate_retrieval(
    retrieved_chunks: list[dict],
    expected_pages: list[int],
) -> dict:
    """Check whether retrieval found any expected source pages."""
    retrieved_pages = {
        int(chunk["page"])
        for chunk in retrieved_chunks
        if "page" in chunk
    }
    expected = {int(page) for page in expected_pages}

    if not expected:
        raise ValueError("expected_pages cannot be empty.")

    matched_pages = retrieved_pages & expected

    return {
        "passed": bool(matched_pages),
        "expected_pages": sorted(expected),
        "retrieved_pages": sorted(retrieved_pages),
        "matched_pages": sorted(matched_pages),
    }


def evaluate_citations(
    answer: str,
    expected_pages: list[int],
) -> dict:
    """
    Check whether the answer cites at least one expected page.
    This checks citation-page matching, not factual support.
    """
    if not expected_pages:
        raise ValueError("expected_pages cannot be empty.")

    cited_pages = {
        int(page)
        for page in re.findall(
            r"\[Page\s+(\d+)\]",
            answer,
            flags=re.IGNORECASE,
        )
    }
    expected = {int(page) for page in expected_pages}
    matched_pages = cited_pages & expected

    return {
        "passed": bool(matched_pages),
        "expected_pages": sorted(expected),
        "cited_pages": sorted(cited_pages),
        "matched_pages": sorted(matched_pages),
    }


def calculate_pass_rate(results: list[dict]) -> float:
    """Return the percentage of test cases that passed."""
    if not results:
        return 0.0

    passed = sum(result["passed"] for result in results)
    return round((passed / len(results)) * 100, 2)
