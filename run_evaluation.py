
import json
from pathlib import Path

from app.ingestion.pdf_parser import extract_text_from_pdf
from app.ingestion.chunking import chunk_pages
from app.retrieval.vector_store import FaissVectorStore
from app.generation.answer_service import answer_question
from app.evaluation.metrics import (
    evaluate_retrieval,
    evaluate_citations,
    calculate_pass_rate,
)


def main():
    dataset_path = Path("data/sample_questions.json")

    if not dataset_path.exists():
        print(f"Dataset not found: {dataset_path}")
        return

    with dataset_path.open("r", encoding="utf-8") as file:
        questions = json.load(file)

    pdf_path = input("Enter the full path to your PDF: ").strip().strip('"')

    try:
        pages = extract_text_from_pdf(pdf_path)
        chunks = chunk_pages(pages, chunk_size=1800, overlap=250)

        store = FaissVectorStore()
        store.add_chunks(chunks)

    except Exception as error:
        print(f"Could not load document: {error}")
        return

    print(f"\nDocument pages: {len(pages)}")
    print(f"Chunks indexed: {store.index.ntotal}")
    print(f"Evaluation questions: {len(questions)}")
    print("\nRunning evaluation...\n")

    retrieval_results = []
    citation_results = []

    for number, item in enumerate(questions, start=1):
        question = item["question"]
        expected_pages = item["expected_pages"]

        print(f"\n{'=' * 60}")
        print(f"Question {number}: {question}")

        try:
            retrieved = store.search(question, top_k=5)

            retrieval = evaluate_retrieval(
                retrieved, expected_pages
            )
            retrieval_results.append(retrieval)

            print(
                f"Retrieval: {'PASS' if retrieval['passed'] else 'FAIL'}"
            )
            print(f"Expected pages: {retrieval['expected_pages']}")
            print(f"Retrieved pages: {retrieval['retrieved_pages']}")

            result = answer_question(store, question, top_k=5)
            citation = evaluate_citations(
                result["answer"], expected_pages
            )
            citation_results.append(citation)

            print(
                f"Citation match: "
                f"{'PASS' if citation['passed'] else 'FAIL'}"
            )
            print(f"Cited pages: {citation['cited_pages']}")
            print(f"Answer: {result['answer'][:500]}")

        except Exception as error:
            print(f"ERROR: {error}")

    print(f"\n{'=' * 60}")
    print("EVALUATION SUMMARY")
    print(f"{'=' * 60}")

    print(
        "Retrieval pass rate: "
        f"{calculate_pass_rate(retrieval_results)}%"
    )
    print(
        "Citation page-match rate: "
        f"{calculate_pass_rate(citation_results)}%"
    )
    print(f"Retrieval tests completed: {len(retrieval_results)}")
    print(f"Citation tests completed: {len(citation_results)}")

    print(
        "\nNote: These are basic checks. A passing citation match "
        "does not guarantee that the answer is factually supported."
    )


if __name__ == "__main__":
    main()
