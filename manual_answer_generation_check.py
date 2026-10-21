
from app.ingestion.pdf_parser import extract_text_from_pdf
from app.ingestion.chunking import chunk_pages
from app.retrieval.vector_store import FaissVectorStore
from app.generation.answer_service import answer_question


def main():
    pdf_path = input("Enter the full path to your PDF: ").strip().strip('"')

    pages = extract_text_from_pdf(pdf_path)
    chunks = chunk_pages(pages, chunk_size=1800, overlap=250)

    store = FaissVectorStore()
    store.add_chunks(chunks)

    print(f"\nDocument loaded: {len(pages)} pages")
    print("ResearchPilot AI is ready!")

    while True:
        question = input(
            "\nAsk a question (or type 'exit'): "
        ).strip()

        if question.lower() == "exit":
            break

        if not question:
            print("Please enter a question.")
            continue

        try:
            result = answer_question(store, question)

            print("\n--- ResearchPilot AI Answer ---")
            print(result["answer"])

            if result["sources"]:
                print("\n--- Retrieved Sources ---")
                for source in result["sources"]:
                    print(
                        f"\nPage {source['page']} | "
                        f"Similarity: {source['similarity']}"
                    )
                    print(source["text"][:300])

        except Exception as error:
            print(f"\nError: {error}")


if __name__ == "__main__":
    main()
