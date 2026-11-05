"""
Formal RAG evaluation.

    python -m app.retrieval.rag_eval \
        --chunks eval/chunks.json --dataset eval/eval_set.json --k 5 \
        --out eval/report.json

chunks.json : list of {"text","page","section","chunk_id"} from your chunker
eval_set.json: see eval_set.example.json

Measures
  Retrieval  Hit@k, Precision@k, Recall@k, MRR, nDCG@k  (hybrid vs +rerank)
  Gate       suggested score threshold separating answerable / unanswerable
  Generation abstention accuracy, citation validity, grounding proxies,
             keyword recall, token-F1 vs reference, tokens, latency
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import re
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


# ---------------------------------------------------------
# Dataset
# ---------------------------------------------------------

@dataclass
class EvalExample:
    id: str
    question: str
    answerable: bool = True
    gold_pages: list[int] = field(default_factory=list)
    gold_chunk_ids: list[Any] = field(default_factory=list)
    gold_phrases: list[str] = field(default_factory=list)   # evidence must contain one
    gold_keywords: list[str] = field(default_factory=list)  # answer should mention
    reference_answer: str | None = None

    @property
    def has_retrieval_labels(self) -> bool:
        return bool(self.gold_pages or self.gold_chunk_ids or self.gold_phrases)


def load_dataset(path: str | Path) -> list[EvalExample]:
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        EvalExample(
            id=str(r.get("id", f"q{i + 1}")),
            question=r["question"],
            answerable=bool(r.get("answerable", True)),
            gold_pages=[int(p) for p in r.get("gold_pages", [])],
            gold_chunk_ids=list(r.get("gold_chunk_ids", [])),
            gold_phrases=list(r.get("gold_phrases", [])),
            gold_keywords=list(r.get("gold_keywords", [])),
            reference_answer=r.get("reference_answer"),
        )
        for i, r in enumerate(rows)
    ]


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).lower())


def _page(chunk: dict[str, Any]) -> int | None:
    try:
        return int(chunk.get("page", chunk.get("page_number")))
    except (TypeError, ValueError):
        return None


def is_relevant(chunk: dict[str, Any], ex: EvalExample) -> bool:
    cid = chunk.get("chunk_id", chunk.get("id"))

    if ex.gold_chunk_ids and cid is not None:
        if str(cid) in {str(x) for x in ex.gold_chunk_ids}:
            return True

    text = _norm(chunk.get("text", ""))
    phrase_ok = (not ex.gold_phrases) or any(_norm(p) in text for p in ex.gold_phrases)

    if ex.gold_pages:
        return _page(chunk) in set(ex.gold_pages) and phrase_ok

    if ex.gold_phrases and not ex.gold_chunk_ids:
        return phrase_ok

    return False


# ---------------------------------------------------------
# Retrieval metrics
# ---------------------------------------------------------

def retrieval_metrics(
    retrieved: list[dict[str, Any]],
    ex: EvalExample,
    all_chunks: list[dict[str, Any]],
    k: int,
) -> dict[str, float | None]:
    flags = [is_relevant(c, ex) for c in retrieved[:k]]
    total_relevant = sum(is_relevant(c, ex) for c in all_chunks)
    hits = sum(flags)

    first = next((i for i, f in enumerate(flags, 1) if f), None)
    dcg = sum(1.0 / math.log2(i + 1) for i, f in enumerate(flags, 1) if f)
    ideal = min(total_relevant, k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal + 1))

    return {
        "hit": 1.0 if hits else 0.0,
        "precision": hits / k,
        "recall": hits / total_relevant if total_relevant else None,
        "mrr": 1.0 / first if first else 0.0,
        "ndcg": dcg / idcg if idcg else None,
    }


def _mean(values: list[float | None]) -> float | None:
    clean = [v for v in values if v is not None]
    return round(statistics.fmean(clean), 4) if clean else None


def suggest_threshold(pos: list[float], neg: list[float]) -> dict[str, float] | None:
    """Score cut-off that best separates answerable from unanswerable questions."""
    if not pos or not neg:
        return None

    best_score, best_t = -1.0, None

    for t in sorted(set(pos + neg)):
        tpr = sum(s >= t for s in pos) / len(pos)
        tnr = sum(s < t for s in neg) / len(neg)
        balanced = (tpr + tnr) / 2

        if balanced > best_score:
            best_score, best_t = balanced, t

    return {"threshold": round(best_t, 4), "balanced_accuracy": round(best_score, 4)}


def _top_score(retrieved: list[dict[str, Any]]) -> float | None:
    if not retrieved:
        return None

    top = retrieved[0]
    value = top.get("reranker_score") if top.get("reranker_used") else top.get("similarity")

    return float(value) if value is not None else None


def evaluate_retrieval(store: Any, examples: list[EvalExample], k: int = 5) -> dict[str, Any]:
    pool = max(k * 4, 20)

    def hybrid(q: str):
        return store.search(q, top_k=k)

    def hybrid_rerank(q: str):
        return store.rerank(q, store.search(q, top_k=pool), top_k=k)

    configs: dict[str, Callable[[str], list[dict[str, Any]]]] = {"hybrid": hybrid}

    if callable(getattr(store, "rerank", None)):
        configs["hybrid+rerank"] = hybrid_rerank

    report: dict[str, Any] = {}

    for name, run in configs.items():
        per, pos, neg, latency, applied = [], [], [], [], []

        for ex in examples:
            t0 = time.perf_counter()
            retrieved = run(ex.question)
            latency.append((time.perf_counter() - t0) * 1000)

            applied.append(bool(retrieved and retrieved[0].get("reranker_used")))
            score = _top_score(retrieved)

            if score is not None:
                (pos if ex.answerable else neg).append(score)

            if ex.answerable and ex.has_retrieval_labels:
                per.append(retrieval_metrics(retrieved, ex, store.chunks, k))

        report[name] = {
            "questions_scored": len(per),
            f"hit@{k}": _mean([m["hit"] for m in per]),
            f"precision@{k}": _mean([m["precision"] for m in per]),
            f"recall@{k}": _mean([m["recall"] for m in per]),
            "mrr": _mean([m["mrr"] for m in per]),
            f"ndcg@{k}": _mean([m["ndcg"] for m in per]),
            "avg_latency_ms": round(statistics.fmean(latency), 1) if latency else None,
            # If this is 0 for the rerank config, the reranker silently did not run.
            "reranker_applied_rate": round(sum(applied) / len(applied), 3) if applied else 0.0,
            "gate_threshold_suggestion": suggest_threshold(pos, neg),
        }

    return report


# ---------------------------------------------------------
# Generation metrics
# ---------------------------------------------------------

_PAGE_RE = re.compile(r"\[\s*pages?\b([^\]]*)\]", re.IGNORECASE)
_ABSTAIN_STATUSES = {"No evidence", "Insufficient evidence", "Withheld", "Error"}


def _terms(text: str) -> set[str]:
    return {
        t[:-1] if t.endswith("s") and len(t) > 4 else t
        for t in re.findall(r"[a-z0-9]+", text.lower())
        if len(t) > 2
    }


def token_f1(prediction: str, reference: str) -> float:
    p, r = _terms(prediction), _terms(reference)

    if not p or not r:
        return 0.0

    overlap = len(p & r)

    if not overlap:
        return 0.0

    precision, recall = overlap / len(p), overlap / len(r)
    return 2 * precision * recall / (precision + recall)


def lexical_grounding(answer: str, evidence: str, threshold: float = 0.6) -> float | None:
    """
    Fraction of answer sentences whose content words appear in the evidence.
    A cheap PROXY for faithfulness; use --llm-judge for a stronger check.
    """
    evidence_terms = _terms(evidence)
    text = _PAGE_RE.sub(" ", answer)
    sentences = [s for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]

    scored = supported = 0

    for sentence in sentences:
        terms = _terms(sentence)

        if len(terms) < 4:
            continue

        scored += 1
        supported += len(terms & evidence_terms) / len(terms) >= threshold

    return supported / scored if scored else None


def cited_pages(answer: str) -> set[int]:
    pages: set[int] = set()

    for match in _PAGE_RE.finditer(answer):
        body = re.split(r"[|;]|section", match.group(1), flags=re.IGNORECASE)[0]

        for a, b in re.findall(r"(\d+)\s*(?:[-–]\s*(\d+))?", body):
            lo, hi = int(a), int(b) if b else int(a)
            if lo <= hi <= lo + 20:
                pages.update(range(lo, hi + 1))

    return pages


def llm_judge(call_llm: Callable[..., str], question: str, answer: str, evidence: str) -> float | None:
    prompt = (
        "You are a strict fact-checker. List each factual claim in the ANSWER "
        "and decide if the EVIDENCE supports it. Reply with JSON only: "
        '{"claims":[{"claim":"...","supported":true}]}\n\n'
        f"QUESTION: {question}\n\nANSWER:\n{answer}\n\nEVIDENCE:\n{evidence[:12000]}"
    )

    try:
        raw = call_llm([{"role": "user", "content": prompt}], max_tokens=800, temperature=0.0)
        claims = json.loads(re.search(r"\{.*\}", raw, re.S).group(0))["claims"]
    except Exception:
        return None

    return sum(bool(c.get("supported")) for c in claims) / len(claims) if claims else None


def evaluate_generation(
    answer_fn: Callable[..., dict[str, Any]],
    store: Any,
    examples: list[EvalExample],
    top_k: int,
    delay: float = 1.0,
    judge: Callable[..., str] | None = None,
) -> dict[str, Any]:
    rows = []

    for ex in examples:
        t0 = time.perf_counter()

        try:
            result = answer_fn(ex.question, store, top_k=top_k)
        except Exception as exc:
            rows.append({"id": ex.id, "error": str(exc)})
            continue

        latency = time.perf_counter() - t0
        answer = str(result.get("answer", ""))
        status = str(result.get("grounding_status", ""))
        evidence = str(result.get("extracted_text", ""))
        source_pages = {s.get("page") for s in result.get("sources", [])}

        abstained = status in _ABSTAIN_STATUSES
        pages = cited_pages(answer)

        row: dict[str, Any] = {
            "id": ex.id,
            "answerable": ex.answerable,
            "status": status,
            "abstained": abstained,
            "correct_behaviour": (not abstained) if ex.answerable else abstained,
            "latency_s": round(latency, 2),
            "total_tokens": (result.get("usage") or {}).get("total_tokens"),
        }

        if ex.answerable and not abstained:
            row["citation_present"] = bool(pages)
            row["citation_valid"] = bool(pages) and pages <= {p for p in source_pages if isinstance(p, int)}
            row["citation_hits_gold"] = bool(pages & set(ex.gold_pages)) if ex.gold_pages else None
            row["lexical_grounding"] = lexical_grounding(answer, evidence)
            row["keyword_recall"] = (
                sum(_norm(k) in _norm(answer) for k in ex.gold_keywords) / len(ex.gold_keywords)
                if ex.gold_keywords else None
            )
            row["token_f1"] = token_f1(answer, ex.reference_answer) if ex.reference_answer else None
            row["llm_faithfulness"] = llm_judge(judge, ex.question, answer, evidence) if judge else None

        rows.append(row)
        time.sleep(delay)

    ok = [r for r in rows if "error" not in r]
    answerable = [r for r in ok if r["answerable"]]
    unanswerable = [r for r in ok if not r["answerable"]]
    answered = [r for r in answerable if not r["abstained"]]

    def rate(items, key):
        values = [r[key] for r in items if r.get(key) is not None]
        return round(sum(map(float, values)) / len(values), 4) if values else None

    summary = {
        "questions": len(rows),
        "errors": len(rows) - len(ok),
        "behaviour_accuracy": rate(ok, "correct_behaviour"),
        "false_abstention_rate": rate(answerable, "abstained"),          # missed answerable
        "hallucination_rate_on_unanswerable": (                           # answered anyway
            round(sum(not r["abstained"] for r in unanswerable) / len(unanswerable), 4)
            if unanswerable else None
        ),
        "citation_present_rate": rate(answered, "citation_present"),
        "citation_valid_rate": rate(answered, "citation_valid"),
        "citation_hits_gold_rate": rate(answered, "citation_hits_gold"),
        "lexical_grounding": rate(answered, "lexical_grounding"),
        "llm_faithfulness": rate(answered, "llm_faithfulness"),
        "keyword_recall": rate(answered, "keyword_recall"),
        "token_f1": rate(answered, "token_f1"),
        "avg_total_tokens": rate(ok, "total_tokens"),
        "avg_latency_s": rate(ok, "latency_s"),
        "status_counts": {s: sum(r.get("status") == s for r in ok) for s in {r.get("status") for r in ok}},
    }

    return {"summary": summary, "rows": rows}


# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------

def _print(title: str, data: dict[str, Any]) -> None:
    print(f"\n== {title} ==")
    for key, value in data.items():
        print(f"  {key:<36} {value}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the RAG pipeline.")
    parser.add_argument("--chunks", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--store-module", default="app.retrieval.vector_store")
    parser.add_argument("--answer-module", default="app.generation.answer_service")
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument("--llm-judge", action="store_true")
    parser.add_argument("--delay", type=float, default=1.0)
    parser.add_argument("--out", default="rag_eval_report.json")
    args = parser.parse_args()

    chunks = json.loads(Path(args.chunks).read_text(encoding="utf-8"))
    examples = load_dataset(args.dataset)

    store = importlib.import_module(args.store_module).FaissVectorStore()
    store.add_chunks(chunks)

    report: dict[str, Any] = {"retrieval": evaluate_retrieval(store, examples, args.k)}

    for name, metrics in report["retrieval"].items():
        _print(f"Retrieval: {name}", metrics)

    if not args.skip_generation:
        module = importlib.import_module(args.answer_module)
        report["generation"] = evaluate_generation(
            module.answer_question, store, examples, args.k, args.delay,
            judge=module._call_groq if args.llm_judge else None,
        )
        _print("Generation", report["generation"]["summary"])

    Path(args.out).write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nFull report written to {args.out}")


if __name__ == "__main__":
    main()