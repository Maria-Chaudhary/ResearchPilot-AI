from conftest import make_store


def top(store, q, k=3):
    return [(r["page"], r["section"]) for r in store.search(q, top_k=k)]


def test_conclusion_heading_found(structured_paper):
    store = make_store(structured_paper)
    assert top(store, "conclusion")[0][1].lower().startswith("conclusion")
    assert top(store, "What is the conclusion of the study?")[0][0] == 10
    assert any(s.lower() == "recommendations" for _, s in top(store, "What are the recommendations?"))


def test_other_intents(structured_paper):
    store = make_store(structured_paper)
    assert top(store, "what is artificial intelligence")[0][0] in (1, 2)
    assert top(store, "Who are the authors?")[0][0] == 1
    assert top(store, "What method was used?")[0][0] == 4
    assert all(sec.lower() != "references" for _, sec in top(store, "conclusion", 2))


def test_conclusion_without_heading_uses_closing_text(essay_paper):
    """Regression: 'conclusion' used to match the word 'conclusions' in mid-paper text."""
    store = make_store(essay_paper)
    results = store.search("what is conclusion of this paper", top_k=3)
    assert results[0]["page"] == 7, [(r["page"], r["section"]) for r in results]
    assert results[0]["section_match"] is True
    assert not results[0]["is_reference"]


def test_index_report(essay_paper):
    report = make_store(essay_paper).index_report()
    assert report["chunks"] == 6 and report["has_conclusion_section"] is False
    assert {s["section"] for s in report["sections"]} >= {"Creativity", "Generalization", "References"}


def test_rerank_degrades_gracefully(structured_paper):
    store = make_store(structured_paper)          # reranker raises -> must not crash
    out = store.rerank("conclusion", store.search("conclusion", top_k=4), top_k=2)
    assert out and all(item["reranker_used"] is False for item in out)


def test_tokenizer_handles_hyphen_and_stems():
    from app.retrieval.vector_store import tokenize
    assert {"self", "attention", "self-attention"} <= set(tokenize("self-attention"))
    assert tokenize("training")[0] == tokenize("trained")[0]