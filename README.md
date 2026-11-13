# ResearchPilot AI 📄🤖

Ask questions about a research paper and get short answers you can trust.

You upload a PDF, type a question, and the app answers in plain language. Every answer shows the **page number** it came from, and the app also shows the **exact lines from the paper** as proof.

If the paper doesn't have the answer, the app says so. It does not make something up.

<!-- Add a screenshot here:
![Demo](docs/demo.png)
-->

## Why I built this

Research papers are long, and finding one answer (like "What is the conclusion?") can take a while. Chatbots can answer quickly, but they sometimes invent things. I wanted a tool that answers from the paper only, and shows where each answer comes from.

## What it can do

- Answer questions about the objectives, methods, findings, limitations, or conclusion
- Write a short summary of the whole paper
- Show page numbers like `[Page 4]` next to each point
- Show proof from the paper, in full sentences
- Say "No evidence" when the paper doesn't answer the question
- Work on papers that have no "Conclusion" heading (it looks at the end of the paper)

## How it works (in simple words)

1. **Read the PDF.** The text is pulled out and cleaned (page numbers and journal headers are removed).
2. **Cut it into pieces.** The text is split into small chunks, always at the end of a sentence, and the app remembers which section each chunk is from.
3. **Find the right pieces.** When you ask a question, the app searches in two ways, by meaning and by keywords. A second model then re-checks which pieces fit best.
4. **Write the answer.** An AI model (Llama, through Groq) writes the answer using only those pieces.
5. **Check the answer.** The app checks that the page numbers, the numbers, and the wording match the paper.
6. **Be honest.** If nothing fits, the app tries once more with more text. If it still finds nothing, it says "No evidence".

## What's inside

```
web_app.py                        the app you open in the browser
app/
├── ingestion/
│   └── chunking.py               cleans the text and cuts it into chunks
├── retrieval/
│   ├── embeddings.py             turns text into numbers for searching
│   ├── vector_store.py           the search part
│   ├── reranker.py               re-checks the best matches
│   └── rag_eval.py               measures how good the results are
└── generation/
    └── answer_service.py         writes and checks the answers
tests/                            automatic tests
```

## How to run it

You need Python 3.10 or newer and a free [Groq API key](https://console.groq.com/).

**1. Make a virtual environment**
```powershell
python -m venv .venv
.venv\Scripts\activate
```

**2. Install the packages**
```powershell
python -m pip install -r requirements.txt
```
(I use `python -m pip` because on some Windows computers `pip.exe` is blocked.)

**3. Add your API key**

Copy `.env.example` to a new file called `.env`, then open it and paste your key:
```
GROQ_API_KEY=your_key_here
```

**4. Start the app**
```powershell
python web_app.py
```

Open the link in your terminal (usually `http://127.0.0.1:7860`), upload a PDF, and ask away.
The first start is slower because the models download once.

## Questions you can try

- `Summarize this paper`
- `What is the conclusion of this paper?`
- `What methodology was used?`
- `What are the main findings?`
- `What are the limitations?`
- `Who are the authors?`

Also try a question the paper can't answer (like "Which GPU was used?"). The app should say "No evidence".

## What the status means

| You see | It means |
|---|---|
| **Partially verified** | The answer passed the page, number, and wording checks |
| **No evidence** | The paper doesn't have enough information |
| **Withheld** | The answer failed the checks, so it was not shown |
| **Summary generated** | A summary of the whole paper |

## Settings (optional)

Put these in your `.env` file if you want to change something:

| Setting | What it does |
|---|---|
| `GROQ_MODEL` | Which Groq model writes the answers |
| `RAG_DISPLAY_EVIDENCE` | How many proof cards to show (default 2) |
| `RAG_DEBUG=1` | Prints what the app found and sent to the AI, for debugging |
| `HF_TOKEN` | Hugging Face token, for faster model downloads |

## Running the tests

```powershell
python -m pip install pytest
python -m pytest
```

The tests don't need internet or an API key. They check the chunking, the search, the answer checks, and error handling.

## Measuring quality

`app/retrieval/rag_eval.py` tests the app with your own questions. It reports how often the right passage is found and how often the app correctly says "No evidence".

```powershell
python -m app.retrieval.rag_eval --chunks chunks.json --dataset eval_set.json --k 5
```

Copy `app/retrieval/eval_set.example.json`, and write 20–30 questions about your paper. Include a few that the paper can't answer.


## Something went wrong?

Turn on debug mode (PowerShell), then ask the same question again:
```powershell
$env:RAG_DEBUG="1"
python web_app.py
```
The terminal will show which passages were found and what the AI replied.

Also, if you change the chunking code, upload the paper again. The old chunks stay in memory until you do.

## Things to know

- The checks compare page numbers, numbers, and wording with the paper. They can't prove that something is true, so always read the evidence cards for important work.
- PDFs with two columns, scans, or big tables can give messy text.
- The free Groq plan has limits, so many questions in a short time may be slowed down.
- Please don't upload private or confidential papers to a shared website.

## Built with

Python · Gradio · FAISS · BM25 · sentence-transformers · Groq (Llama 3.3) · PyMuPDF · pytest

## License

Add a license file before sharing (MIT is a common simple choice).
