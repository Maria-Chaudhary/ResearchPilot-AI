import pytest

from conftest import make_store

EVIDENCE = [
    {"page": 7, "text": "We propose a two-stage method. Accuracy reached 92.3% on the benchmark."},
    {"page": 5, "text": "Training used the SciDocs dataset containing 1,200 papers collected from arXiv."},
]


def script(ag, monkeypatch, replies, record=None):
    """Replace the network call with scripted replies (str or callable)."""
    it = iter(replies)

    def fake(messages, max_tokens, temperature):
        if record is not None:
            record.append(messages[-1]["content"])
        reply = next(it)
        return (reply(messages) if callable(reply) else reply), "stop", {"total_tokens": 100}

    monkeypatch.setattr(ag, "_groq_request", fake)


@pytest.mark.parametrize("answer", [
    "The method has two stages [Page 7]. Accuracy reached 92.3% [Page 7].\n1. Encode\n2. Classify",
    "Accuracy was 92.3% [Page 7 | Section: Results].",
    "Data came from 1,200 papers [Pages 5, 7].",
    "About 92% accuracy was reported [Page 7].",
])
def test_valid_answers_accepted(ag, answer):
    ok, why = ag._validate_answer(answer, EVIDENCE)
    assert ok, why


@pytest.mark.parametrize("answer", [
    "Accuracy was 97.1% [Page 7].",     # number not in evidence
    "It works [Page 99].",              # page not supplied
    "It works well.",                   # no citation
])
def test_invalid_answers_rejected(ag, answer):
    assert not ag._validate_answer(answer, EVIDENCE)[0]


def test_citation_normalisation(ag):
    assert ag._normalize_citations("x [Page 7 | Section: Results] y [Pages 5-6]") == "x [Page 7] y [Pages 5, 6]"


@pytest.mark.parametrize("question,expected", [
    ("What is the overall accuracy?", "results"),
    ("Summarize the methodology", "methodology"),
    ("Summarize this paper", "summary"),
    ("what is this document about", "summary"),
    ("What is the conclusion of this paper?", "conclusion"),
])
def test_question_routing(ag, question, expected):
    assert ag._classify_question(question) == expected


def test_conclusion_without_heading_end_to_end(ag, monkeypatch, essay_paper):
    """The user's failing case: conclusion text must reach the model and be cited."""
    store = make_store(essay_paper)
    prompts = []
    script(ag, monkeypatch, [
        "The paper concludes that the central goal of AI is to build programs that learn from experience and generalise from prior cases [Page 7].",
    ], record=prompts)

    result = ag.answer_question("what is conclusion of this paper", store, top_k=3)

    assert result["grounding_status"] == "Partially verified"
    assert "[Page 7]" in result["answer"]
    assert "In summary, the central goal of AI" in prompts[0]
    assert "QUESTION TYPE:\nconclusion" in prompts[0]
    assert result["sources"] and result["sources"][0]["page"] == 7
    assert result["sources"][0]["text"].startswith("In summary")


def test_honest_abstention_not_forced_to_guess(ag, monkeypatch, structured_paper):
    store = make_store(structured_paper)
    prompts = []
    script(ag, monkeypatch, ["INSUFFICIENT_EVIDENCE", "INSUFFICIENT_EVIDENCE"], record=prompts)
    result = ag.answer_question("What GPU was used for training?", store, top_k=1)
    assert result["grounding_status"] == "No evidence"
    assert "does not contain enough" in result["answer"]
    assert not any("failed an automatic check" in p for p in prompts)


def test_second_chance_widens_evidence(ag, monkeypatch, structured_paper):
    store = make_store(structured_paper)
    calls = []
    script(ag, monkeypatch, [
        "INSUFFICIENT_EVIDENCE",
        "The study found that artificial intelligence benefits the human race [Page 10].",
    ], record=calls)
    result = ag.answer_question("What did the study find about benefits?", store, top_k=1)
    assert result["retrieval"]["second_chance"] is True
    assert "Be thorough" in calls[1]
    assert result["grounding_status"] == "Partially verified"


def test_validation_retry_fixes_hallucinated_number(ag, monkeypatch, structured_paper):
    store = make_store(structured_paper)
    script(ag, monkeypatch, [
        "The study used 5000 respondents [Page 4].",
        "The study used 400 respondents [Page 4].",
    ])
    result = ag.answer_question("How many respondents were used?", store, top_k=3)
    assert result["grounding_status"] == "Partially verified"
    assert "400" in result["answer"]


def test_unsupported_sentence_is_flagged(ag):
    passages = [{"page": 10, "text": "Findings revealed that artificial intelligence benefits the human race."}]
    clean = ag._grounding_check("Artificial intelligence benefits the human race. [Page 10]", passages)
    bad = ag._grounding_check("The study proves AI will replace all teachers within a decade. [Page 10]", passages)
    assert clean["unsupported"] == [] and len(bad["unsupported"]) == 1


def test_broad_question_summarises_without_retrieval(ag, monkeypatch):
    class Store:
        chunks = [{"text": "I am a motivated Generative AI Developer with hands-on experience building AI applications using Python and RAG.", "page": 1, "section": "General", "chunk_id": 0}]

        def search(self, *a, **k):
            raise AssertionError("summary must not use retrieval")

    calls = []
    script(ag, monkeypatch, ["This is the CV of a Generative AI Developer experienced in Python and RAG [Page 1]."], record=calls)
    result = ag.answer_question("what is this document about", Store())
    assert len(calls) == 1 and result["grounding_status"] == "Summary generated"


def test_truncated_answer_is_continued(ag, monkeypatch):
    parts = iter([("Training used the SciDocs", "length"), (" dataset [Page 5].", "stop")])
    monkeypatch.setattr(ag, "_groq_request", lambda m, mt, t: (*next(parts), {}))
    assert ag._call_groq([{"role": "user", "content": "x"}]) == "Training used the SciDocs dataset [Page 5]."


def test_evidence_cards_are_complete_sentences(ag):
    broken = "ided by the following objectives: To find out the benefits. The study employed the survey method in which questionnaires were distributed. Data was analysed using the in understanding the"
    shown = ag._complete_sentences(broken)
    assert shown.startswith("The study employed") and shown.endswith("distributed.")


def test_api_errors_never_crash_the_ui(ag, monkeypatch, structured_paper):
    store = make_store(structured_paper)

    def boom(*a, **k):
        raise RuntimeError("Groq API returned HTTP 503: unavailable")

    monkeypatch.setattr(ag, "_groq_request", boom)
    result = ag.answer_question("What method was used?", store)
    assert result["grounding_status"] == "No evidence" and "503" in result["answer"]


def test_input_limits(ag, structured_paper):
    store = make_store(structured_paper)
    assert "too long" in ag.answer_question("x" * 5000, store)["answer"]
    assert "enter a question" in ag.answer_question("   ", store)["answer"]
    assert "upload a PDF" in ag.answer_question("hello", None)["answer"]
    with pytest.raises(ValueError):
        ag.answer_question("hello", store, top_k=99)


def test_health_check_reports_without_raising(ag, structured_paper):
    report = ag.health_check(make_store(structured_paper))
    assert report["ok"] is True and report["groq_api_key_set"] is True
    assert report["index"]["chunks"] == len(structured_paper)
    assert "reranker_available" in report


def test_markdown_artefacts_removed_from_evidence(ag):
    raw = "## Creativity\n**==> picture [10 x 10] omitted <==**\nThis **important** claim is a full sentence about creativity. A second complete sentence follows here."
    shown = ag._complete_sentences(raw)
    assert "*" not in shown and "#" not in shown and "picture" not in shown
    assert shown.startswith("This important claim")