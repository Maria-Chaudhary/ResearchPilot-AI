
from app.ingestion.pdf_parser import extract_text_from_pdf

pdf_path = input("Enter the full path to your PDF: ").strip().strip('"')

try:
    pages = extract_text_from_pdf(pdf_path)

    print(f"\nSuccessfully extracted {len(pages)} pages.")

    for item in pages[:3]:
        print(f"\n--- Page {item['page']} ---")
        print(item["text"][:500])

except (FileNotFoundError, ValueError) as error:
    print(f"Error: {error}")
