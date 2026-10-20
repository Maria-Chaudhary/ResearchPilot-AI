
import gradio as gr

from app.ingestion.pdf_parser import extract_text_from_pdf
from app.ingestion.chunking import chunk_pages
from app.retrieval.embeddings import embed_texts
from app.retrieval.vector_store import FaissVectorStore
from app.generation.answer_service import answer_question


# Keep the embedding model loaded between uploads.

def process_pdf(pdf_file):
    if pdf_file is None:
        return None, "Please upload a PDF first."

    try:
        pages = extract_text_from_pdf(pdf_file)
        
        chunks = chunk_pages(
           pages,
          chunk_size=1000,
           overlap=150,
)

        store = FaissVectorStore()
        store.add_chunks(chunks)

        status = (
            "PDF processed successfully!\n"
            f"Pages extracted: {len(pages)}\n"
            f"Chunks indexed: {len(chunks)}"
        )

        return store, status

    except Exception as exc:
        return None, f"Could not process PDF: {exc}"
    
def ask_question(question, store):
    """Answer a question using the indexed document."""

    if store is None:
        return "Please upload and process a PDF first.", ""

    if not question or not question.strip():
        return "Please enter a question.", ""

    try:
        result = answer_question(store, question, top_k=3)

        answer = result["answer"]
        sources = result.get("sources", [])

        if sources:
            source_text = "\n\n".join(
                (
                    f"Page {source['page']} "
                    f"(similarity: {source['similarity']:.3f})\n"
                    f"{source['text']}"
                )
                for source in sources
            )
        else:
            source_text = "No supporting passages were returned."

        return answer, source_text

    except Exception as exc:
        return f"Error: {exc}", ""


with gr.Blocks(title="ResearchPilot AI") as demo:
    gr.Markdown(
        """
        # ResearchPilot AI
        ### Ask questions about your research papers and PDFs.

        Upload a document, process it, and ask questions.
        Answers are generated using retrieved document passages.
        """
    )

    document_store = gr.State(value=None)

    with gr.Row():
        pdf_input = gr.File(
            label="Upload a PDF",
            file_types=[".pdf"],
            type="filepath",
        )

    process_button = gr.Button(
        "Process PDF",
        variant="primary",
    )
    status_output = gr.Textbox(
        label="Document Status",
        lines=4,
        interactive=False,
    )

    gr.Markdown("## Ask a question")

    question_input = gr.Textbox(
        label="Your question",
        placeholder="What are the main findings of this paper?",
        lines=2,
    )

    ask_button = gr.Button("Ask ResearchPilot")
    answer_output = gr.Markdown(label="Answer")
    sources_output = gr.Textbox(
        label="Retrieved Source Passages",
        lines=12,
        interactive=False,
    )

    process_button.click(
        fn=process_pdf,
        inputs=[pdf_input],
        outputs=[document_store, status_output],
    )

    ask_button.click(
        fn=ask_question,
        inputs=[question_input, document_store],
        outputs=[answer_output, sources_output],
    )

    question_input.submit(
        fn=ask_question,
        inputs=[question_input, document_store],
        outputs=[answer_output, sources_output],
    )


if __name__ == "__main__":
    demo.launch()
