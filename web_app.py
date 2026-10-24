import gradio as gr

from app.ingestion.pdf_parser import extract_text_from_pdf
from app.ingestion.chunking import chunk_pages
from app.retrieval.vector_store import FaissVectorStore
from app.generation.answer_service import answer_question


# ─────────────────────────────────────────────
# ResearchPilot AI — Research Paper Assistant
# ─────────────────────────────────────────────

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --rp-bg: #f6f4fb;
    --rp-bg-2: #efeafc;
    --rp-card: #ffffff;
    --rp-text: #1a1533;
    --rp-muted: #6b6588;
    --rp-primary: #6d28d9;
    --rp-primary-2: #8b5cf6;
    --rp-primary-3: #a78bfa;
    --rp-accent: #c026d3;
    --rp-border: #e8e2f7;
    --rp-border-strong: #d6c9f5;
    --rp-shadow: 0 4px 24px rgba(88, 45, 180, 0.06);
    --rp-shadow-lg: 0 20px 50px rgba(88, 45, 180, 0.12);
}

* {
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
}

.gradio-container {
    font-family: 'DM Sans', sans-serif !important;
    background:
        radial-gradient(1200px 600px at 10% -10%, #ede4ff 0%, transparent 55%),
        radial-gradient(1000px 500px at 100% 0%, #f3e8ff 0%, transparent 50%),
        var(--rp-bg) !important;
    max-width: 1280px !important;
    margin: auto !important;
    padding: 28px 24px 10px !important;
}

.gradio-container h1,
.gradio-container h2,
.gradio-container h3 {
    font-family: 'Manrope', sans-serif !important;
    color: var(--rp-text) !important;
    letter-spacing: -0.4px;
}

/* ── Header ─────────────────────────────── */
#rp-header {
    position: relative;
    background:
        radial-gradient(900px 400px at 85% -30%, rgba(192, 38, 211, 0.55) 0%, transparent 60%),
        radial-gradient(700px 500px at 0% 120%, rgba(139, 92, 246, 0.55) 0%, transparent 55%),
        linear-gradient(135deg, #1e1147 0%, #3b1d8f 45%, #6d28d9 100%);
    border-radius: 28px;
    padding: 44px 48px 42px;
    color: white;
    margin-bottom: 26px;
    overflow: hidden;
    box-shadow: 0 30px 60px -20px rgba(76, 29, 149, 0.55);
    isolation: isolate;
}

#rp-header::before {
    content: "";
    position: absolute;
    inset: 0;
    background-image:
        linear-gradient(rgba(255, 255, 255, 0.045) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255, 255, 255, 0.045) 1px, transparent 1px);
    background-size: 42px 42px;
    mask-image: radial-gradient(ellipse at 30% 0%, black 20%, transparent 75%);
    -webkit-mask-image: radial-gradient(ellipse at 30% 0%, black 20%, transparent 75%);
    pointer-events: none;
    z-index: -1;
}

#rp-header::after {
    content: "";
    position: absolute;
    width: 320px;
    height: 320px;
    right: -90px;
    top: -110px;
    background: radial-gradient(circle, rgba(167, 139, 250, 0.55), transparent 65%);
    filter: blur(10px);
    z-index: -1;
}

#rp-header h1 {
    color: #ffffff !important;
    font-size: clamp(30px, 4.4vw, 46px) !important;
    font-weight: 800 !important;
    letter-spacing: -1.6px;
    margin: 0 0 12px !important;
    background: linear-gradient(100deg, #ffffff 30%, #e9d5ff 70%, #f0abfc 100%);
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
}

#rp-header p {
    color: #ddd2fb !important;
    font-size: 15.5px;
    line-height: 1.75;
    max-width: 640px;
    margin: 0;
}

.rp-eyebrow {
    display: inline-block;
    color: #e9d5ff;
    text-transform: uppercase;
    letter-spacing: 2.4px;
    font-size: 11px;
    font-weight: 700;
    margin-bottom: 16px;
    padding: 6px 14px;
    border-radius: 999px;
    background: rgba(167, 139, 250, 0.16);
    border: 1px solid rgba(196, 181, 253, 0.35);
    backdrop-filter: blur(6px);
}

/* ── Cards ──────────────────────────────── */
.rp-card {
    background: var(--rp-card);
    border: 1px solid var(--rp-border);
    border-radius: 22px;
    padding: 26px 26px 24px;
    box-shadow: var(--rp-shadow);
    transition: box-shadow 0.28s ease, border-color 0.28s ease, transform 0.28s ease;
    position: relative;
    overflow: hidden;
}

.rp-card::before {
    content: "";
    position: absolute;
    inset: 0 0 auto 0;
    height: 3px;
    background: linear-gradient(90deg, var(--rp-primary-2), var(--rp-accent), var(--rp-primary));
    opacity: 0.85;
}

.rp-card:hover {
    box-shadow: var(--rp-shadow-lg);
    border-color: var(--rp-border-strong);
    transform: translateY(-2px);
}

.rp-card h3 {
    font-size: 20px !important;
    font-weight: 700 !important;
    margin: 2px 0 8px !important;
}

.rp-section-label {
    display: inline-block;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 1.6px;
    font-weight: 500;
    color: var(--rp-primary);
    margin-bottom: 10px;
    padding: 4px 10px;
    border-radius: 7px;
    background: linear-gradient(120deg, #f3ecff 0%, #fbefff 100%);
    border: 1px solid #eadcff;
}

.rp-muted {
    color: var(--rp-muted);
    font-size: 13.5px;
    line-height: 1.7;
}

/* ── Buttons ────────────────────────────── */
#rp-process-btn,
#rp-ask-btn {
    min-height: 50px;
    border-radius: 14px !important;
    font-weight: 700 !important;
    font-size: 14.5px !important;
    letter-spacing: 0.2px;
    border: none !important;
    transition: transform 0.18s ease, box-shadow 0.18s ease, filter 0.18s ease !important;
    position: relative;
    overflow: hidden;
}

#rp-process-btn {
    background: linear-gradient(135deg, #7c3aed 0%, #6d28d9 55%, #5b21b6 100%) !important;
    color: #fff !important;
    box-shadow: 0 10px 24px -8px rgba(109, 40, 217, 0.65) !important;
}

#rp-ask-btn {
    background: linear-gradient(135deg, #3b1d8f 0%, #2a1566 100%) !important;
    color: #fff !important;
    box-shadow: 0 10px 24px -8px rgba(42, 21, 102, 0.55) !important;
}

#rp-process-btn:hover,
#rp-ask-btn:hover {
    transform: translateY(-2px);
    filter: brightness(1.06);
}

#rp-process-btn:active,
#rp-ask-btn:active {
    transform: translateY(0);
}

button {
    transition: transform 0.18s ease, box-shadow 0.18s ease !important;
    font-weight: 600 !important;
}

button:hover {
    transform: translateY(-1px);
}

/* Secondary / clear buttons */
.gradio-container button.secondary,
.gradio-container .secondary {
    border-radius: 14px !important;
    border: 1px solid var(--rp-border-strong) !important;
    background: #fbf9ff !important;
    color: var(--rp-primary) !important;
    min-height: 50px;
    font-weight: 600 !important;
}

.gradio-container button.secondary:hover,
.gradio-container .secondary:hover {
    background: #f3ecff !important;
    border-color: var(--rp-primary-3) !important;
}

/* ── Inputs / textareas ─────────────────── */
textarea, input[type="text"], input[type="number"] {
    border-radius: 13px !important;
    border-color: var(--rp-border) !important;
    background: #fdfcff !important;
    font-size: 14px !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
}

textarea:focus, input[type="text"]:focus {
    border-color: var(--rp-primary-2) !important;
    box-shadow: 0 0 0 4px rgba(139, 92, 246, 0.14) !important;
    background: #ffffff !important;
}

/* Labels */
.gradio-container label > span,
.gradio-container .label-wrap > span {
    font-weight: 600 !important;
    color: #4b4568 !important;
    font-size: 12.5px !important;
    letter-spacing: 0.3px;
}

/* ── File uploader ──────────────────────── */
.gradio-container .file-preview,
.gradio-container [data-testid="block-label"] {
    border-radius: 14px !important;
}

.gradio-container .upload-container,
.gradio-container .file-upload {
    border: 2px dashed var(--rp-border-strong) !important;
    border-radius: 16px !important;
    background: linear-gradient(180deg, #fbf8ff 0%, #f7f2ff 100%) !important;
    transition: border-color 0.22s ease, background 0.22s ease !important;
}

.gradio-container .upload-container:hover,
.gradio-container .file-upload:hover {
    border-color: var(--rp-primary-2) !important;
    background: #f5eeff !important;
}

/* ── Answer output ──────────────────────── */
#rp-answer {
    background:
        linear-gradient(180deg, #fdfcff 0%, #faf6ff 100%);
    border: 1px solid var(--rp-border);
    border-left: 4px solid var(--rp-primary-2);
    border-radius: 16px;
    padding: 22px 24px;
    min-height: 180px;
    line-height: 1.75;
    color: #2a2344;
    font-size: 14.5px;
    box-shadow: inset 0 1px 2px rgba(109, 40, 217, 0.03);
}

#rp-answer p {
    color: #2a2344 !important;
}

/* ── Status + sources textareas ─────────── */
#rp-status textarea {
    border-radius: 13px !important;
    line-height: 1.7 !important;
    background: linear-gradient(180deg, #fbf9ff 0%, #f8f4ff 100%) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 12.5px !important;
    color: #3a3159 !important;
    border-color: var(--rp-border) !important;
}

#rp-sources textarea {
    border-radius: 13px !important;
    line-height: 1.75 !important;
    background: linear-gradient(180deg, #fcfaff 0%, #faf6ff 100%) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 12.5px !important;
    color: #3a3159 !important;
    border-color: var(--rp-border) !important;
}

/* ── Scrollbar ──────────────────────────── */
::-webkit-scrollbar {
    width: 10px;
    height: 10px;
}

::-webkit-scrollbar-track {
    background: #f2ecff;
    border-radius: 10px;
}

::-webkit-scrollbar-thumb {
    background: linear-gradient(180deg, #c4b5fd, #a78bfa);
    border-radius: 10px;
    border: 2px solid #f2ecff;
}

::-webkit-scrollbar-thumb:hover {
    background: linear-gradient(180deg, #a78bfa, #8b5cf6);
}

/* ── Footer ─────────────────────────────── */
footer {
    display: none !important;
}

/* ── Responsive ─────────────────────────── */
@media (max-width: 700px) {
    .gradio-container {
        padding: 16px 12px 6px !important;
    }

    #rp-header {
        padding: 30px 24px;
        border-radius: 22px;
    }

    .rp-card {
        padding: 20px 18px;
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
        primary_hue="purple",
        secondary_hue="violet",
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
        <div style="text-align:center; padding:22px 8px; color:#7b6ba8; font-size:13px; letter-spacing:0.3px;">
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