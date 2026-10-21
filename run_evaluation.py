
import argparse
import json
import sys
from pathlib import Path

from app.ingestion.pdf_parser import extract_text_from_pdf
from app.ingestion.chunking import chunk_pages
from app.retrieval.vector_store import FaissVectorStore


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET = PROJECT_ROOT / "data" / "evaluation_dataset.json"


def load_dataset(dataset_path: Path) -> list[dict]:
    """Load and validate the evaluation questions."""

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Evaluation dataset not found: {dataset_path}\n"
            "Create data/evaluation_dataset.json first."
        )

    with dataset_path.open("r", encoding="utf-8") as file:
        dataset = json.load(file)

    if not isinstance(dataset, list) or not dataset:
        raise ValueError(
            "The evaluation dataset must be a non-empty JSON list."
        )

    for number, item in enumerate(dataset, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Question {number} must be a JSON object.")

        question = item.get("question")
        relevant_pages = item.get("relevant_pages")

        if not isinstance(question, str) or not question.strip():
            raise ValueError(
                f"Question {number} needs a non-empty 'question' field."
            )

        if not isinstance(relevant_pages, list) or not relevant_pages:
            raise ValueError(
                f"Question {number} needs a non-empty 'relevant_pages' list."
            )

        for page in relevant_pages:
            if (
                not isinstance(page, int)
                or isinstance(page, bool)
                or page < 1
            ):
                raise ValueError(
                    f"Question {number} has an invalid relevant page: {page}"
                )

    return dataset


def calculate_metrics(
    retrieved_pages: list[int],
    relevant_pages: list[int],
) -> tuple[float, float, bool]:
    """
    Calculate page-based Recall@K, reciprocal rank, and hit status.

    Recall@K = relevant pages retrieved / all relevant pages.
    Reciprocal rank = 1 / rank of the first relevant result.
    """

    relevant = set(relevant_pages)
    retrieved_relevant = set(retrieved_pages) & relevant

    recall = len(retrieved_relevant) / len(relevant)

    reciprocal_rank = 0.0

    for rank, page in enumerate(retrieved_pages, start=1):
        if page in relevant:
            reciprocal_rank = 1.0 / rank
            break

    hit = any(page in relevant for page in retrieved_pages)

    return recall, reciprocal_rank, hit


def evaluate(
    pdf_path: Path,
    dataset_path: Path,
    top_k: int = 3,
    chunk_size: int = 1200,
    overlap: int = 200,
) -> dict:
    """Build the research-paper index and evaluate retrieval."""

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError("The input document must be a PDF.")

    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    print("\n" + "=" * 60)
    print("          RESEARCHPILOT AI - RETRIEVAL EVALUATION")
    print("=" * 60)

    print(f"\nPDF: {pdf_path.name}")
    print(f"Dataset: {dataset_path}")
    print(f"Top K: {top_k}")

    # 1. Load evaluation questions
    dataset = load_dataset(dataset_path)
    print(f"\nEvaluation questions: {len(dataset)}")

    # 2. Extract text from the research paper
    print("\n[1/3] Extracting text from PDF...")
    pages = extract_text_from_pdf(str(pdf_path))

    print(f"Extracted text from {len(pages)} pages.")

    # 3. Split extracted text into passages
    print("\n[2/3] Creating research-paper chunks...")
    chunks = chunk_pages(
        pages,
        chunk_size=chunk_size,
        overlap=overlap,
    )

    if not chunks:
        raise ValueError("No text chunks were created from this PDF.")

    print(f"Created {len(chunks)} chunks.")

    # 4. Create the vector index
    print("\n[3/3] Building FAISS vector index...")
    store = FaissVectorStore()
    store.add_chunks(chunks)

    print(f"Indexed {len(store.chunks)} chunks.")
    print("\nRunning retrieval tests...\n")

    # 5. Evaluate every question
    question_results = []

    for number, item in enumerate(dataset, start=1):
        question = item["question"].strip()
        relevant_pages = item["relevant_pages"]

        retrieved = store.search(question, top_k=top_k)

        retrieved_pages = [
            result["page"]
            for result in retrieved
        ]

        recall, reciprocal_rank, hit = calculate_metrics(
            retrieved_pages,
            relevant_pages,
        )

        result = {
            "question": question,
            "relevant_pages": relevant_pages,
            "retrieved_pages": retrieved_pages,
            "recall_at_k": recall,
            "reciprocal_rank": reciprocal_rank,
            "hit": hit,
            "retrieved_chunks": retrieved,
        }

        question_results.append(result)

        status = "HIT" if hit else "MISS"

        print(f"Question {number}: {question}")
        print(f"  Expected relevant pages: {relevant_pages}")
        print(f"  Retrieved pages:         {retrieved_pages}")
        print(f"  Recall@{top_k}:               {recall:.3f}")
        print(f"  Reciprocal rank:         {reciprocal_rank:.3f}")
        print(f"  Result:                  {status}")
        print("-" * 60)

    # 6. Calculate overall metrics
    question_count = len(question_results)

    mean_recall = sum(
        result["recall_at_k"] for result in question_results
    ) / question_count

    mean_reciprocal_rank = sum(
        result["reciprocal_rank"] for result in question_results
    ) / question_count

    hit_rate = sum(
        result["hit"] for result in question_results
    ) / question_count

    summary = {
        "pdf": pdf_path.name,
        "dataset": str(dataset_path),
        "top_k": top_k,
        "number_of_questions": question_count,
        "number_of_chunks": len(chunks),
        "recall_at_k": round(mean_recall, 4),
        "mean_reciprocal_rank": round(mean_reciprocal_rank, 4),
        "hit_rate": round(hit_rate, 4),
        "results": [
            {
                "question": result["question"],
                "relevant_pages": result["relevant_pages"],
                "retrieved_pages": result["retrieved_pages"],
                "recall_at_k": round(result["recall_at_k"], 4),
                "reciprocal_rank": round(
                    result["reciprocal_rank"], 4
                ),
                "hit": result["hit"],
            }
            for result in question_results
        ],
    }

    # 7. Display the summary
    print("\n" + "=" * 60)
    print("                 FINAL RESULTS")
    print("=" * 60)
    print(f"Questions evaluated: {question_count}")
    print(f"Recall@{top_k}:            {mean_recall:.2%}")
    print(f"Mean Reciprocal Rank: {mean_reciprocal_rank:.2%}")
    print(f"Hit Rate:              {hit_rate:.2%}")
    print("=" * 60)

    # 8. Save results to a JSON report
    report_path = PROJECT_ROOT / "evaluation_results.json"

    with report_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    print(f"\nDetailed report saved to:\n{report_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate ResearchPilot AI's retrieval quality "
            "against known relevant PDF pages."
        )
    )

    parser.add_argument(
        "pdf",
        help="Path to the research-paper PDF.",
    )

    parser.add_argument(
        "--dataset",
        default=str(DEFAULT_DATASET),
        help="Path to the evaluation JSON dataset.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of passages to retrieve per question (default: 3).",
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1200,
        help="Maximum chunk size in characters (default: 1200).",
    )

    parser.add_argument(
        "--overlap",
        type=int,
        default=200,
        help="Chunk overlap in characters (default: 200).",
    )

    args = parser.parse_args()

    try:
        evaluate(
            pdf_path=Path(args.pdf).expanduser(),
            dataset_path=Path(args.dataset).expanduser(),
            top_k=args.top_k,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
        )

    except (FileNotFoundError, ValueError, RuntimeError) as error:
        print(f"\nEvaluation error: {error}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
