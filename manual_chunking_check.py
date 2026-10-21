
from app.ingestion.pdf_parser import extract_text_from_pdf
from app.ingestion.chunking import chunk_pages

pdf_path = input("Enter the full path to your PDF: ").strip().strip('"')

try:
    # Step 1: Extract text from the PDF.
    pages = extract_text_from_pdf(pdf_path)

    # Step 2: Split the extracted text into chunks.
    chunks = chunk_pages(pages, chunk_size=1800, overlap=250)

    print(f"\nPDF pages extracted: {len(pages)}")
    print(f"Text chunks created: {len(chunks)}")

    for chunk in chunks[:5]:
        print(f"\n--- Chunk {chunk['chunk_id']} | Page {chunk['page']} ---")
        print(chunk["text"][:300])

except (FileNotFoundError, ValueError) as error:
    print(f"Error: {error}")
