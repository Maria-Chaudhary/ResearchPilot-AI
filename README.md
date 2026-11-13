# ResearchPilot AI

Upload a research paper (PDF), ask a question, and get a short answer with **page citations**. The app also shows the exact passages from the paper that support the answer.

If the paper does not contain the answer, it says so instead of guessing.

<!-- Add a screenshot or GIF here:
![ResearchPilot AI demo](docs/demo.png)
-->

## Features

- **Ask anything about one paper:** objectives, methods, findings, limitations, conclusion, authors, or a short summary.
- **Page citations:** every claim is marked like `[Page 4]`.
- **Evidence cards:** complete sentences from the paper, with page number and section.
- **Honest answers:** if the evidence is missing, the app answers "No evidence".
- **Answer checks:** page numbers, numbers and wording are compared with the evidence.
- **Works on papers with or without headings:** a paper that has no "Conclusion" section still gets its closing text found.

## How it works

```
PDF
 └─ text extraction (PyMuPDF)
     └─ chunking          sentence-based chunks, section tracking, header/footer removal
         └─ indexing      embeddings in FAISS + BM25 keyword index
             └─ question
                 ├─ broad question ("summarize this paper") → whole-document summary
                 └─ focused question
                     ├─ hybrid search (FAISS + BM25, Reciprocal Rank Fusion)
                     ├─ section / position boost (e.g. conclusion → closing text)
                     ├─ cross-encoder reranking
                     ├─ token-budgeted context
                     ├─ LLM answer (Llama via Groq), using only the evidence
                     ├─ checks: citations, numbers, sentence-level wording
                     └─ one wider second attempt before saying "No evidence"
```

## Project structure

```
web_app.py                         Gradio app
app/
├── ingestion/
│   └── chunking.py                chunking, headings (including Markdown), header removal
├── retrieval/
│   ├── embeddings.py              embedding model
│   ├── vector_store.py            FAISS + BM25 search, section/position boost, rerank()
│   ├── reranker.py                cross-encoder reranker
│   └── rag_eval.py                evaluation script
└── generation/
    └── answer_service.py          answer_question(), checks, summaries, health_check()
tests/                             automated tests (no model downloads needed)
```

## Setup

You need Python 3.10 or newer and a free [Groq API key](https://console.groq.com/).

```powershell
# 1. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Mac / Linux

# 2. Install packages
python -m pip install -r requirements.txt

# 3. Add your API key
copy .env.example .env          # then open .env and set GROQ_API_KEY
```

> If `pip.exe` is blocked on your computer, always use `python -m pip ...` as above.

## Run

```powershell
python web_app.py
```

Open the link shown in the terminal (usually `http://127.0.0.1:7860`), upload a PDF, and ask a question.

The first start downloads the embedding and reranker models, so it takes longer.

## Example questions

| Question | What you get |
|---|---|
| `Summarize this paper` | A short summary with page citations |
| `What is the conclusion of this paper?` | The authors' concluding statements |
| `What methodology was used?` | The methods or research design section |
| `What are the main findings?` | Results with page citations |
| `What are the limitations?` | The limitations, or "No evidence" if the paper has none |
| `Who are the authors?` | Title and author information from page 1 |

## Reading the result

| Status | Meaning |
|---|---|
| Partially verified | The answer passed the citation, number and wording checks |
| No evidence | The paper does not contain enough information |
| Withheld | The answer failed the checks and was not shown |
| Summary generated | A whole-document summary |

A more detailed label (`Verified` or `Partially verified`) and any unsupported sentences are in `result["grounding_detail"]`.

## Configuration

Set these in `.env` (all optional except `GROQ_API_KEY`):

| Variable | Default | Purpose |
|---|---|---|
| `GROQ_API_KEY` | none | Your Groq API key |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Model used for answers |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Reranking model |
| `HF_TOKEN` | none | Hugging Face token (faster model downloads) |
| `RAG_CONTEXT_WINDOW_TOKENS` | `8192` | Context size used for budgeting |
| `RAG_MAX_CONTEXT_TOKENS` | `5000` | Maximum evidence tokens per question |
| `RAG_DISPLAY_EVIDENCE` | `2` | Evidence cards shown |
| `RAG_STRICT_GATE` | `0` | `1` also requires similarity thresholds before asking the model |
| `RAG_DEBUG` | `0` | `1` prints a retrieval trace |
| `RAG_LOG_QUESTIONS` | `0` | `1` adds the question text to logs (off for privacy) |

## Tests

```powershell
python -m pip install pytest
python -m pytest
```

The tests use small stand-ins for the embedding model and the Groq API, so they run offline and fast. They cover chunking, retrieval, answer checks, error handling and empty model replies.

## Evaluation

`app/retrieval/rag_eval.py` measures retrieval quality (Hit@k, Precision@k, Recall@k, MRR, nDCG@k) and answer behavior (abstention, hallucination rate, citation validity, grounding).

```powershell
python -m app.retrieval.rag_eval --chunks chunks.json --dataset eval_set.json --k 5
```

Create `eval_set.json` from `app/retrieval/eval_set.example.json` with 20–30 questions about your own paper, including a few the paper cannot answer.

Results on my test set: **[add your own numbers here]**

## Debugging a bad answer

Start the app with debug output (PowerShell):

```powershell
$env:RAG_DEBUG="1"
python web_app.py
```

The terminal then shows the retrieved passages, their scores, the passages sent to the model, and the model's raw replies.

## Limitations

- The checks compare citations, numbers and wording with the evidence. They do **not** prove that a statement is true.
- PDFs with two columns, scans or complex tables can produce messy text. This comes from text extraction, not from the chunking.
- Short answers depend on the quality of the chunks. After changing the chunking code, upload the paper again.
- The free Groq tier has rate limits.
- Do not upload confidential documents to a shared deployment.

## Tech stack

Python · Gradio · FAISS · BM25 · sentence-transformers (embeddings and cross-encoder reranking) · Groq API (Llama 3.3) · PyMuPDF · pytest

## License

Add a license before sharing (for example MIT) and put it in a `LICENSE` file.
