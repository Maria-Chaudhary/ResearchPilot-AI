
from app.ingestion.pdf_parser import extract_text_from_pdf
from app.ingestion.chunking import chunk_pages
from app.retrieval.vector_store import FaissVectorStore


def main():
    pdf_path = input("Enter the full path to your research PDF: ")
    pdf_path = pdf_path.strip().strip('"')

    # 1. Extract PDF text.
    pages = extract_text_from_pdf(pdf_path)
    print(f"\nPages extracted: {len(pages)}")

    # 2. Split the text into chunks.
    chunks = chunk_pages(pages, chunk_size=1800, overlap=250)
    print(f"Chunks created: {len(chunks)}")

    # 3. Create embeddings and index the chunks.
    store = FaissVectorStore()
    store.add_chunks(chunks)
    print(f"Chunks indexed in FAISS: {store.index.ntotal}")

    # 4. Search the paper.
    while True:
        query = input(
            "\nAsk a question about the paper (or type 'exit'): "
        ).strip()

        if query.lower() == "exit":
            break

        if not query:
            print("Please enter a question.")
            continue

        results = store.search(query, top_k=5)

        print("\nMost relevant passages:")

        for rank, item in enumerate(results, start=1):
            print(
                f"\n--- Result {rank} | "
                f"Page {item['page']} | "
                f"Similarity {item['similarity']:.4f} ---"
            )
            print(item["text"][:700])


if __name__ == "__main__":
    main()
