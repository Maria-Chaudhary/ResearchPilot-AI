
import gradio as gr

from app.ingestion.pdf_parser import extract_text_from_pdf
from app.ingestion.chunking import chunk_pages
from app.retrieval.vector_store import FaissVectorStore
from app.generation.answer_service import answer_question


# ─────────────────────────────────────────────
# ResearchPilot AI — Research Paper Assistant
# ─────────────────────────────────────────────

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

:root {
    --rp-bg: #f5f7fb;
    --rp-card: #ffffff;
    --rp-text: #182235;
    --rp-muted: #69758a;
    --rp-primary: #5b5bd6;
    --rp-border: #e5e9f2;
}

.gradio-container {
    font-family: 'DM Sans', sans-serif !important;
    background: var(--rp-bg) !important;
    max-width: 1250px !important;
    margin: auto !important;
}

.gradio-container h1,
.gradio-container h2,
.gradio-container h3 {
    font-family: 'Manrope', sans-serif !important;
    color: var(--rp-text) !important;
}

#rp-header {
    background: linear-gradient(125deg, #20254b 0%, #4545a8 58%, #7774e8 100%);
    border-radius: 24px;
    padding: 34px 36px;
    color: white;
    margin-bottom: 22px;
    box-shadow: 0 14px 35px rgba(55, 55, 135, 0.15);
}

#rp-header h1 {
    color: white !important;
    font-size: clamp(28px, 4vw, 42px) !important;
    font-weight: 800 !important;
    letter-spacing: -1.4px;
    margin-bottom: 8px;
}

#rp-header p {
    color: #e4e5ff !important;
    font-size: 15px;
    line-height: 1.7;
}

.rp-eyebrow {
    color: #c8caff;
    text-transform: uppercase;
    letter-spacing: 2px;
    font-size: 11px;
    font-weight: 700;
    margin-bottom: 12px;
}

.rp-card {
    background: var(--rp-card);
    border: 1px solid var(--rp-border);
    border-radius: 19px;
    padding: 20px;
    box-shadow: 0 5px 18px rgba(35, 48, 80, 0.035);
}

.rp-section-label {
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 1.3px;
    font-weight: 700;
    color: var(--rp-primary);
    margin-bottom: 7px;
}

.rp-muted {
    color: var(--rp-muted);
    font-size: 13px;
    line-height: 1.65;
}

#rp-process-btn {
    min-height: 48px;
    border-radius: 12px !important;
    font-weight: 700 !important;
    background: #5b5bd6 !important;
    border: none !important;
}

#rp-ask-btn {
    min-height: 48px;
    border-radius: 12px !important;
    font-weight: 700 !important;
    background: #252b50 !important;
    border: none !important;
}

#rp-answer {
    background: white;
    border: 1px solid var(--rp-border);
    border-radius: 16px;
    padding: 20px;
    min-height: 180px;
}

#rp-status textarea,
#rp-sources textarea {
    border-radius: 12px !important;
    line-height: 1.65 !important;
}

textarea, input {
    border-radius: 11px !important;
}

button {
    transition: transform 0.18s ease, box-shadow 0.18s ease !important;
}

button:hover {
    transform: translateY(-1px);
}

footer {
    display: none !important;
}

@media (max-width: 700px) {
    #rp-header {
        padding: 24px 20px;
    }

    .rp-card {
        padding: 15px;
    }
}
"""


def process_pdf(pdf_file):
    if pdf_file is None:
        return None, "Please upload a research paper first."

    try:
        pages = extract_text_from_pdf(pdf_file)

        # Research-focused chunking settings
        chunks = chunk_pages(
            pages,
            chunk_size=1200,
            overlap=200,
        )

        if not chunks:
            return None, "No usable text chunks were found in this PDF."

        store = FaissVectorStore()
        store.add_chunks(chunks)

        status = (
            "✓ Research paper processed successfully!\n\n"
            f"Pages extracted: {len(pages)}\n"
            f"Research passages indexed: {len(chunks)}\n\n"
            "Your paper is ready for questions."
        )

        return store, status

    except Exception as exc:
        return None, f"Could not process the PDF: {exc}"


def ask_question(question, store):
    if store is None:
        return (
            "### Upload a paper first\n"
            "Upload your research PDF and select **Process Paper**.",
            "",
        )

    if not question or not question.strip():
        return "Please enter a research question.", ""

    try:
        result = answer_question(store, question.strip(), top_k=3)

        answer = result.get(
            "answer",
            "No answer was returned.",
        )
        sources = result.get("sources", [])

        if sources:
            source_parts = []

            for source in sources:
                page = source.get("page", "?")
                similarity = source.get("similarity", 0.0)
                text = source.get("text", "").strip()

                max_chars = 700
                if len(text) > max_chars:
                    text = text[:max_chars].rsplit(" ", 1)[0] + "..."

                source_parts.append(
                    f"PAGE {page}  ·  Similarity: {similarity:.3f}\n"
                    f"{text}"
                )

            source_text = "\n\n────────────────────────\n\n".join(
                source_parts
            )
        else:
            source_text = (
                "No supporting passages were returned. "
                "Try rephrasing your question."
            )

        return answer, source_text

    except Exception as exc:
        return f"**Something went wrong:** {exc}", ""


# ─────────────────────────────────────────────
# User Interface
# ─────────────────────────────────────────────

with gr.Blocks(
    theme=gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="slate",
        neutral_hue="slate",
        radius_size="lg",
        spacing_size="md",
    ),
    css=CUSTOM_CSS,
    title="ResearchPilot AI | Research Assistant",
) as demo:

    gr.HTML(
        """
        <div id="rp-header">
            <div class="rp-eyebrow">AI-powered research workspace</div>
            <h1>ResearchPilot AI</h1>
            <p>
                Read smarter. Explore research faster.
                Upload an academic paper, ask questions about its findings,
                and inspect the supporting passages.
            </p>
        </div>
        """
    )

    document_store = gr.State(value=None)

    with gr.Row(equal_height=True):

        # LEFT: Document workspace
        with gr.Column(scale=4, min_width=280):
            with gr.Group(elem_classes=["rp-card"]):
                gr.HTML(
                    """
                    <div class="rp-section-label">01 / Your document</div>
                    <h3>Research library</h3>
                    <p class="rp-muted">
                        Upload a text-based PDF of a research paper.
                        Its extracted passages will be indexed for Q&A.
                    </p>
                    """
                )

                pdf_input = gr.File(
                    label="Upload research paper",
                    file_types=[".pdf"],
                    type="filepath",
                )

                process_button = gr.Button(
                    "Process Paper  →",
                    variant="primary",
                    elem_id="rp-process-btn",
                )

                status_output = gr.Textbox(
                    label="Processing status",
                    value="Waiting for a research paper...",
                    lines=5,
                    interactive=False,
                    elem_id="rp-status",
                )

        # RIGHT: Research Q&A
        with gr.Column(scale=7, min_width=300):
            with gr.Group(elem_classes=["rp-card"]):
                gr.HTML(
                    """
                    <div class="rp-section-label">02 / Research assistant</div>
                    <h3>Ask your paper</h3>
                    <p class="rp-muted">
                        Ask about the research problem, methodology,
                        dataset, findings, limitations, or conclusion.
                    </p>
                    """
                )

                question_input = gr.Textbox(
                    label="Your research question",
                    placeholder=(
                        "Example: What methodology does this paper use?"
                    ),
                    lines=3,
                )

                with gr.Row():
                    ask_button = gr.Button(
                        "Find Answer  →",
                        variant="primary",
                        elem_id="rp-ask-btn",
                    )
                    clear_button = gr.Button(
                        "Clear question",
                        variant="secondary",
                    )

                gr.Markdown("### Answer", elem_classes=["rp-section-label"])

                answer_output = gr.Markdown(
                    value=(
                        "*Your answer will appear here after you process "
                        "a paper and ask a question.*"
                    ),
                    elem_id="rp-answer",
                )

    # Evidence area
    with gr.Group(elem_classes=["rp-card"]):
        gr.HTML(
            """
            <div class="rp-section-label">03 / Evidence and citations</div>
            <h3>Retrieved passages</h3>
            <p class="rp-muted">
                Inspect the text retrieved from your paper to check
                what supports the generated answer.
            </p>
            """
        )

        sources_output = gr.Textbox(
            label="Source passages and page numbers",
            placeholder="Supporting evidence will appear here...",
            lines=12,
            interactive=False,
            elem_id="rp-sources",
        )

    gr.Markdown(
        """
        <div style="text-align:center; padding:22px 8px; color:#7b8497;">
            ResearchPilot AI · Evidence-grounded academic reading
        </div>
        """
    )

    # Actions
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

    clear_button.click(
        fn=lambda: ("",),
        inputs=[],
        outputs=[question_input],
    )


if __name__ == "__main__":
    demo.launch()
