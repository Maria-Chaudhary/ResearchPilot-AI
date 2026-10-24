import html

import gradio as gr

import re
from app.ingestion.pdf_parser import extract_text_from_pdf
from app.ingestion.chunking import chunk_pages
from app.retrieval.vector_store import FaissVectorStore
from app.generation.answer_service import answer_question


# ─────────────────────────────────────────────
# ResearchPilot AI — Research Paper Assistant
# ─────────────────────────────────────────────

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Figtree:wght@400;500;600;700&family=Newsreader:opsz,wght@6..72,500;6..72,600;6..72,700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ═════════ Design tokens ═════════ */
:root {
    --rp-bg: #f4f5fb;
    --rp-card: #ffffff;
    --rp-card-soft: #f8f8fe;
    --rp-ink: #12142a;
    --rp-text: #2b2f4a;
    --rp-muted: #646a86;
    --rp-line: #e4e6f2;
    --rp-line-strong: #cfd3ea;
    --rp-indigo: #4f46e5;
    --rp-indigo-deep: #3730a3;
    --rp-violet: #7c3aed;
    --rp-mark: #fbbf24;
    --rp-ok: #10b981;
    --rp-shadow: 0 1px 2px rgba(18, 20, 42, 0.04), 0 8px 28px rgba(55, 48, 163, 0.07);
    --rp-shadow-lg: 0 2px 4px rgba(18, 20, 42, 0.05), 0 24px 56px rgba(55, 48, 163, 0.16);
    --rp-ease: cubic-bezier(0.22, 1, 0.36, 1);
}

.dark {
    --rp-bg: #0d0f1e;
    --rp-card: #151833;
    --rp-card-soft: #1b1f3f;
    --rp-ink: #eef0ff;
    --rp-text: #c9cdea;
    --rp-muted: #9399bd;
    --rp-line: #262a4d;
    --rp-line-strong: #363b69;
    --rp-shadow: 0 1px 2px rgba(0, 0, 0, 0.3), 0 8px 28px rgba(0, 0, 0, 0.35);
    --rp-shadow-lg: 0 2px 4px rgba(0, 0, 0, 0.4), 0 24px 56px rgba(0, 0, 0, 0.5);
}

* {
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
}

html {
    scroll-behavior: smooth;
}

.gradio-container {
    font-family: 'Figtree', system-ui, sans-serif !important;
    background:
        radial-gradient(1100px 520px at 8% -8%, rgba(99, 102, 241, 0.16) 0%, transparent 60%),
        radial-gradient(900px 480px at 100% 0%, rgba(167, 139, 250, 0.14) 0%, transparent 55%),
        var(--rp-bg) !important;
    max-width: 1240px !important;
    margin: auto !important;
    padding: 28px 24px 12px !important;
}

.gradio-container h1,
.gradio-container h2,
.gradio-container h3 {
    font-family: 'Newsreader', Georgia, serif !important;
    color: var(--rp-ink) !important;
    letter-spacing: -0.3px;
}

/* ═════════ Keyframes ═════════ */
@keyframes rp-rise {
    from { opacity: 0; transform: translateY(18px); }
    to   { opacity: 1; transform: translateY(0); }
}

@keyframes rp-fade {
    from { opacity: 0; }
    to   { opacity: 1; }
}

@keyframes rp-drift-a {
    0%, 100% { transform: translate3d(0, 0, 0) scale(1); }
    50%      { transform: translate3d(-60px, 40px, 0) scale(1.15); }
}

@keyframes rp-drift-b {
    0%, 100% { transform: translate3d(0, 0, 0) scale(1); }
    50%      { transform: translate3d(70px, -30px, 0) scale(1.1); }
}

@keyframes rp-float {
    0%, 100% { transform: rotate(2.5deg) translateY(0); }
    50%      { transform: rotate(1.5deg) translateY(-10px); }
}

@keyframes rp-scan {
    0%   { top: -40px; opacity: 0; }
    10%  { opacity: 1; }
    90%  { opacity: 1; }
    100% { top: 100%; opacity: 0; }
}

@keyframes rp-mark {
    0%, 18%  { background-size: 0% 100%; }
    38%, 78% { background-size: 100% 100%; }
    100%     { background-size: 0% 100%; }
}

@keyframes rp-pulse {
    0%   { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.55); }
    70%  { box-shadow: 0 0 0 9px rgba(16, 185, 129, 0); }
    100% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
}

@keyframes rp-grow {
    from { width: 0; }
}

@keyframes rp-shimmer {
    0%   { background-position: -200% 0; }
    100% { background-position: 200% 0; }
}

/* ═════════ Header (the one orchestrated moment) ═════════ */
#rp-header {
    position: relative;
    border-radius: 28px;
    padding: 48px 52px;
    margin-bottom: 26px;
    color: #fff;
    overflow: hidden;
    isolation: isolate;
    background: linear-gradient(135deg, #0f1130 0%, #1e1b6e 48%, #4338ca 100%);
    box-shadow: 0 34px 70px -24px rgba(55, 48, 163, 0.65);
    animation: rp-fade 0.7s var(--rp-ease) both;
}

#rp-header::before,
#rp-header::after {
    content: "";
    position: absolute;
    border-radius: 50%;
    filter: blur(60px);
    z-index: -1;
    pointer-events: none;
}

#rp-header::before {
    width: 420px;
    height: 420px;
    right: -80px;
    top: -160px;
    background: rgba(167, 139, 250, 0.75);
    animation: rp-drift-a 14s ease-in-out infinite;
}

#rp-header::after {
    width: 380px;
    height: 380px;
    left: -100px;
    bottom: -190px;
    background: rgba(56, 189, 248, 0.45);
    animation: rp-drift-b 17s ease-in-out infinite;
}

.rp-hero {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 44px;
}

.rp-hero-copy {
    flex: 1 1 auto;
    min-width: 0;
}

.rp-brand-pill {
    display: inline-flex;
    align-items: center;
    gap: 10px;
    font-size: 13px;
    font-weight: 600;
    color: #e0e3ff;
    padding: 7px 15px 7px 12px;
    border-radius: 999px;
    background: rgba(255, 255, 255, 0.1);
    border: 1px solid rgba(255, 255, 255, 0.22);
    backdrop-filter: blur(8px);
    margin-bottom: 20px;
    animation: rp-rise 0.8s var(--rp-ease) 0.1s both;
}

.rp-live-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #34d399;
    animation: rp-pulse 2.2s ease-out infinite;
}

#rp-header h1 {
    font-size: clamp(34px, 5vw, 56px) !important;
    font-weight: 600 !important;
    line-height: 1.04 !important;
    letter-spacing: -1.4px !important;
    margin: 0 0 16px !important;
    color: #ffffff !important;
    animation: rp-rise 0.9s var(--rp-ease) 0.2s both;
}

#rp-header p {
    color: #cfd3f7 !important;
    font-size: 16.5px;
    line-height: 1.7;
    max-width: 520px;
    margin: 0;
    animation: rp-rise 0.9s var(--rp-ease) 0.32s both;
}

/* Animated paper: highlights + scan beam = "finding evidence" */
.rp-doc {
    position: relative;
    flex: 0 0 auto;
    width: 240px;
    padding: 26px 24px 18px;
    border-radius: 18px;
    background: rgba(255, 255, 255, 0.09);
    border: 1px solid rgba(255, 255, 255, 0.22);
    backdrop-filter: blur(10px);
    box-shadow: 0 30px 60px -20px rgba(8, 6, 40, 0.6);
    overflow: hidden;
    animation: rp-rise 1s var(--rp-ease) 0.45s both, rp-float 7s ease-in-out 1.5s infinite;
}

.rp-doc .rp-doc-title {
    display: block;
    height: 10px;
    width: 62%;
    border-radius: 5px;
    background: rgba(255, 255, 255, 0.55);
    margin-bottom: 18px;
}

.rp-doc .ln {
    display: block;
    height: 7px;
    border-radius: 4px;
    margin-bottom: 11px;
    background-color: rgba(255, 255, 255, 0.24);
}

.rp-doc .ln.hit {
    background-image: linear-gradient(90deg, var(--rp-mark), #fde68a);
    background-repeat: no-repeat;
    background-size: 0% 100%;
    animation: rp-mark 6s ease-in-out infinite;
}

.rp-doc .ln.hit.d2 { animation-delay: 1.6s; }
.rp-doc .ln.hit.d3 { animation-delay: 3.2s; }

.rp-doc .w100 { width: 100%; }
.rp-doc .w90  { width: 90%; }
.rp-doc .w80  { width: 80%; }
.rp-doc .w70  { width: 70%; }
.rp-doc .w55  { width: 55%; }

.rp-beam {
    position: absolute;
    left: 0;
    right: 0;
    height: 44px;
    top: -44px;
    background: linear-gradient(180deg, transparent, rgba(196, 181, 253, 0.55), transparent);
    animation: rp-scan 6s ease-in-out infinite;
    pointer-events: none;
}

/* ═════════ Cards ═════════ */
.rp-card {
    --block-background-fill: transparent;
    --block-border-width: 0px;
    --block-shadow: none;
    --input-background-fill: var(--rp-card-soft);
    background: var(--rp-card) !important;
    border: 1px solid var(--rp-line) !important;
    border-radius: 22px !important;
    padding: 28px !important;
    box-shadow: var(--rp-shadow);
    position: relative;
    overflow: hidden;
    transition: box-shadow 0.4s var(--rp-ease), border-color 0.4s var(--rp-ease), transform 0.4s var(--rp-ease);
    animation: rp-rise 0.8s var(--rp-ease) both;
}

.rp-card .styler,
.rp-card > div {
    background: transparent !important;
}

.rp-card::before {
    content: "";
    position: absolute;
    inset: 0 0 auto 0;
    height: 3px;
    background: linear-gradient(90deg, var(--rp-indigo), var(--rp-violet), #38bdf8);
    transform: scaleX(0.18);
    transform-origin: left;
    transition: transform 0.7s var(--rp-ease);
}

.rp-card:hover,
.rp-card:focus-within {
    box-shadow: var(--rp-shadow-lg);
    border-color: var(--rp-line-strong) !important;
    transform: translateY(-3px);
}

.rp-card:hover::before,
.rp-card:focus-within::before {
    transform: scaleX(1);
}

#rp-card-library  { animation-delay: 0.5s; }
#rp-card-ask      { animation-delay: 0.62s; }
#rp-card-evidence { animation-delay: 0.74s; }

/* Step heading (these three steps really are a sequence) */
.rp-step {
    display: flex;
    gap: 16px;
    align-items: flex-start;
    margin-bottom: 8px;
}

.rp-num {
    flex: 0 0 auto;
    width: 34px;
    height: 34px;
    border-radius: 11px;
    display: grid;
    place-items: center;
    font-family: 'Newsreader', Georgia, serif;
    font-weight: 700;
    font-size: 17px;
    color: #fff;
    background: linear-gradient(135deg, var(--rp-indigo), var(--rp-violet));
    box-shadow: 0 8px 18px -6px rgba(79, 70, 229, 0.6);
    transition: transform 0.45s var(--rp-ease);
}

.rp-card:hover .rp-num {
    transform: rotate(-8deg) scale(1.08);
}

.rp-step h3 {
    font-size: 23px !important;
    font-weight: 600 !important;
    margin: 0 0 4px !important;
    line-height: 1.2 !important;
}

.rp-muted {
    color: var(--rp-muted);
    font-size: 14px;
    line-height: 1.65;
    margin: 0;
}

.rp-sub {
    font-family: 'Newsreader', Georgia, serif !important;
    font-size: 17px;
    font-weight: 600;
    color: var(--rp-ink);
    margin: 18px 0 8px;
}

/* ═════════ Buttons ═════════ */
button {
    font-family: 'Figtree', sans-serif !important;
    font-weight: 600 !important;
}

#rp-process-btn,
#rp-ask-btn {
    min-height: 52px;
    border-radius: 14px !important;
    font-size: 15px !important;
    letter-spacing: 0.1px;
    border: none !important;
    color: #fff !important;
    position: relative;
    overflow: hidden;
    transition: transform 0.3s var(--rp-ease), box-shadow 0.3s var(--rp-ease), filter 0.3s ease !important;
}

#rp-process-btn {
    background: linear-gradient(135deg, #6366f1 0%, #4f46e5 55%, #4338ca 100%) !important;
    box-shadow: 0 12px 26px -10px rgba(79, 70, 229, 0.8) !important;
}

#rp-ask-btn {
    background: linear-gradient(135deg, #7c3aed 0%, #5b21b6 100%) !important;
    box-shadow: 0 12px 26px -10px rgba(91, 33, 182, 0.75) !important;
}

#rp-process-btn::after,
#rp-ask-btn::after {
    content: "";
    position: absolute;
    inset: 0;
    background: linear-gradient(110deg, transparent 30%, rgba(255, 255, 255, 0.35) 50%, transparent 70%);
    transform: translateX(-120%);
    transition: transform 0.8s var(--rp-ease);
    pointer-events: none;
}

#rp-process-btn:hover,
#rp-ask-btn:hover {
    transform: translateY(-3px);
    filter: brightness(1.07);
}

#rp-process-btn:hover::after,
#rp-ask-btn:hover::after {
    transform: translateX(120%);
}

#rp-process-btn:active,
#rp-ask-btn:active {
    transform: translateY(0) scale(0.98);
}

.gradio-container button.secondary {
    min-height: 52px;
    border-radius: 14px !important;
    border: 1px solid var(--rp-line-strong) !important;
    background: var(--rp-card-soft) !important;
    color: var(--rp-indigo) !important;
    transition: transform 0.3s var(--rp-ease), background 0.25s ease, border-color 0.25s ease !important;
}

.gradio-container button.secondary:hover {
    background: var(--rp-card) !important;
    border-color: var(--rp-indigo) !important;
    transform: translateY(-2px);
}

.gradio-container button.secondary:active {
    transform: scale(0.98);
}

/* ═════════ Inputs ═════════ */
textarea,
input[type="text"],
input[type="number"] {
    border-radius: 13px !important;
    border-color: var(--rp-line) !important;
    background: var(--rp-card-soft) !important;
    color: var(--rp-ink) !important;
    font-family: 'Figtree', sans-serif !important;
    font-size: 15px !important;
    transition: border-color 0.25s ease, box-shadow 0.25s ease, background 0.25s ease !important;
}

textarea:focus,
input[type="text"]:focus {
    border-color: var(--rp-indigo) !important;
    box-shadow: 0 0 0 4px rgba(79, 70, 229, 0.15) !important;
    background: var(--rp-card) !important;
}

.gradio-container label > span,
.gradio-container .label-wrap > span {
    font-weight: 600 !important;
    color: var(--rp-muted) !important;
    font-size: 13px !important;
}

/* ═════════ File uploader ═════════ */
.gradio-container .upload-container,
.gradio-container .file-upload,
.gradio-container [data-testid="file-upload"] {
    border: 2px dashed var(--rp-line-strong) !important;
    border-radius: 16px !important;
    background: var(--rp-card-soft) !important;
    transition: border-color 0.3s ease, background 0.3s ease, transform 0.3s var(--rp-ease) !important;
}

.gradio-container .upload-container:hover,
.gradio-container .file-upload:hover,
.gradio-container [data-testid="file-upload"]:hover {
    border-color: var(--rp-indigo) !important;
    background: rgba(99, 102, 241, 0.07) !important;
    transform: scale(1.005);
}

/* ═════════ Status box ═════════ */
#rp-status textarea {
    line-height: 1.75 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 12.5px !important;
    color: var(--rp-text) !important;
    border-left: 3px solid var(--rp-ok) !important;
}

/* ═════════ Answer ═════════ */
#rp-answer {
    background: var(--rp-card-soft);
    border: 1px solid var(--rp-line);
    border-left: 4px solid var(--rp-indigo);
    border-radius: 16px;
    padding: 22px 26px;
    min-height: 170px;
    line-height: 1.8;
    color: var(--rp-text);
    font-size: 15.5px;
    transition: border-color 0.4s ease, box-shadow 0.4s ease;
}

#rp-answer:hover {
    border-left-color: var(--rp-violet);
    box-shadow: 0 10px 30px -14px rgba(79, 70, 229, 0.3);
}

#rp-answer p,
#rp-answer li {
    color: var(--rp-text) !important;
}

#rp-answer strong {
    color: var(--rp-ink);
}

#rp-answer h3 {
    font-size: 21px !important;
    margin: 0 0 6px !important;
}

#rp-answer .prose > *,
#rp-answer .md > * {
    animation: rp-rise 0.6s var(--rp-ease) both;
}

#rp-answer .prose > *:nth-child(2) { animation-delay: 0.08s; }
#rp-answer .prose > *:nth-child(3) { animation-delay: 0.16s; }
#rp-answer .prose > *:nth-child(4) { animation-delay: 0.24s; }
#rp-answer .prose > *:nth-child(n+5) { animation-delay: 0.32s; }

/* ═════════ Evidence cards ═════════ */
.rp-evidence {
    display: flex;
    flex-direction: column;
    gap: 14px;
    margin-top: 6px;
}

.rp-src {
    position: relative;
    padding: 18px 20px 18px 22px;
    border-radius: 16px;
    background: var(--rp-card-soft);
    border: 1px solid var(--rp-line);
    overflow: hidden;
    transition: transform 0.35s var(--rp-ease), box-shadow 0.35s var(--rp-ease), border-color 0.35s ease;
    animation: rp-rise 0.6s var(--rp-ease) both;
    animation-delay: calc(var(--i, 0) * 0.1s);
}

.rp-src::before {
    content: "";
    position: absolute;
    left: 0;
    top: 0;
    bottom: 0;
    width: 4px;
    background: linear-gradient(180deg, var(--rp-mark), #f59e0b);
}

.rp-src:hover {
    transform: translateX(4px);
    border-color: var(--rp-line-strong);
    box-shadow: 0 14px 34px -16px rgba(79, 70, 229, 0.35);
}

.rp-src-head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 16px;
    flex-wrap: wrap;
    margin-bottom: 12px;
}

.rp-page {
    font-weight: 700;
    font-size: 13px;
    color: var(--rp-indigo-deep);
    background: rgba(99, 102, 241, 0.12);
    padding: 5px 12px;
    border-radius: 8px;
}

.dark .rp-page {
    color: #c7d2fe;
}

.rp-meter {
    display: flex;
    align-items: center;
    gap: 10px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    color: var(--rp-muted);
}

.rp-meter-track {
    width: 110px;
    height: 6px;
    border-radius: 999px;
    background: var(--rp-line);
    overflow: hidden;
}

.rp-meter-fill {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, var(--rp-indigo), var(--rp-violet));
    animation: rp-grow 1.1s var(--rp-ease) 0.3s both;
}

.rp-src-text {
    margin: 0;
    font-size: 14px;
    line-height: 1.75;
    color: var(--rp-text);
}

.rp-empty {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 22px;
    border-radius: 16px;
    border: 2px dashed var(--rp-line-strong);
    color: var(--rp-muted);
    font-size: 14px;
    line-height: 1.6;
    background: linear-gradient(110deg, transparent 30%, rgba(99, 102, 241, 0.06) 50%, transparent 70%);
    background-size: 200% 100%;
    animation: rp-shimmer 5s linear infinite;
}

.rp-empty svg {
    flex: 0 0 auto;
}

/* ═════════ Gradio progress / loading ═════════ */
.eta-bar {
    background: linear-gradient(90deg, rgba(99, 102, 241, 0.25), rgba(124, 58, 237, 0.25)) !important;
}

.generating {
    border-color: var(--rp-indigo) !important;
}

/* ═════════ Footer ═════════ */
.rp-footer {
    text-align: center;
    padding: 26px 8px 14px;
    color: var(--rp-muted);
    font-size: 13px;
    animation: rp-fade 1s ease 1s both;
}

footer {
    display: none !important;
}

/* ═════════ Scrollbar ═════════ */
::-webkit-scrollbar { width: 10px; height: 10px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb {
    background: var(--rp-line-strong);
    border-radius: 10px;
    border: 2px solid transparent;
    background-clip: padding-box;
}
::-webkit-scrollbar-thumb:hover { background-color: var(--rp-indigo); }

/* ═════════ Accessibility ═════════ */
button:focus-visible,
textarea:focus-visible,
input:focus-visible {
    outline: 3px solid rgba(99, 102, 241, 0.55) !important;
    outline-offset: 2px;
}

@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after {
        animation-duration: 0.01ms !important;
        animation-iteration-count: 1 !important;
        transition-duration: 0.01ms !important;
    }
}

/* ═════════ Responsive ═════════ */
@media (max-width: 860px) {
    .rp-doc { display: none; }
    #rp-header { padding: 34px 26px; border-radius: 22px; }
}

@media (max-width: 700px) {
    .gradio-container { padding: 16px 12px 6px !important; }
    .rp-card { padding: 20px 18px !important; border-radius: 18px !important; }
    .rp-step h3 { font-size: 20px !important; }
    .rp-meter-track { width: 70px; }
}
"""


HEADER_HTML = """
<div id="rp-header">
    <div class="rp-hero">
        <div class="rp-hero-copy">
            <div class="rp-brand-pill">
                <span class="rp-live-dot"></span>
                AI research workspace
            </div>
            <h1>ResearchPilot AI</h1>
            <p>
                Upload an academic paper, ask about its findings, and check every
                answer against the exact passages it came from.
            </p>
        </div>
        <div class="rp-doc" aria-hidden="true">
            <span class="rp-beam"></span>
            <span class="rp-doc-title"></span>
            <span class="ln w100"></span>
            <span class="ln hit w90"></span>
            <span class="ln w80"></span>
            <span class="ln w100"></span>
            <span class="ln hit d2 w70"></span>
            <span class="ln w90"></span>
            <span class="ln w55"></span>
            <span class="ln w100"></span>
            <span class="ln hit d3 w80"></span>
            <span class="ln w70"></span>
        </div>
    </div>
</div>
"""

LIBRARY_HTML = """
<div class="rp-step">
    <div class="rp-num">1</div>
    <div>
        <h3>Add your paper</h3>
        <p class="rp-muted">
            Upload a text-based PDF. Its passages are indexed so you can search them with questions.
        </p>
    </div>
</div>
"""

ASK_HTML = """
<div class="rp-step">
    <div class="rp-num">2</div>
    <div>
        <h3>Ask your paper</h3>
        <p class="rp-muted">
            Ask about the research problem, methodology, dataset, findings, limitations, or conclusion.
        </p>
    </div>
</div>
"""

EVIDENCE_HTML = """
<div class="rp-step">
    <div class="rp-num">3</div>
    <div>
        <h3>Check the evidence</h3>
        <p class="rp-muted">
            These are the passages the answer is based on, with their page numbers.
        </p>
    </div>
</div>
"""

EMPTY_SOURCES_HTML = """
<div class="rp-empty">
    <svg width="34" height="34" viewBox="0 0 24 24" fill="none" stroke="currentColor"
         stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">
        <path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z"/>
        <path d="M14 3v5h5"/>
        <path d="M9 13h6M9 17h4"/>
    </svg>
    <div>Supporting passages will show up here after you ask a question.</div>
</div>
"""

FOOTER_HTML = """
<div class="rp-footer">
    ResearchPilot AI &nbsp;—&nbsp; answers grounded in your paper
</div>
"""

ANSWER_PLACEHOLDER = (
    "*Your answer will appear here after you process a paper and ask a question.*"
)
def build_sources_html(sources):
    """Render retrieved passages as readable evidence cards."""
    cards = []

    for index, source in enumerate(sources):
        page = html.escape(str(source.get("page", "?")))
        section = html.escape(str(source.get("section", "Unknown")))

        try:
            similarity = float(source.get("similarity", 0.0))
        except (TypeError, ValueError):
            similarity = 0.0

        text = str(source.get("text", "") or "").strip()

        # Normalize whitespace without damaging word boundaries.
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text).strip()

        # Limit display length, preferring a sentence boundary.
        max_chars = 1200

        if len(text) > max_chars:
            clipped = text[:max_chars]
            sentence_ends = list(
                re.finditer(r'[.!?]["\')\]]?(?=\s|$)', clipped)
            )

            if sentence_ends and sentence_ends[-1].end() >= max_chars * 0.5:
                text = clipped[:sentence_ends[-1].end()].strip()
            else:
                last_space = clipped.rfind(" ")
                text = (
                    clipped[:last_space].rstrip()
                    if last_space > 0
                    else clipped.rstrip()
                )
                text += " …"

        safe_text = html.escape(text).replace("\n", "<br>")

        # Similarity must be supplied by answer_service.py to be meaningful.
        percent = max(0.0, min(1.0, similarity)) * 100

        cards.append(
            f"""
            <div class="rp-src" style="--i:{index}">
                <div class="rp-src-head">
                    <span class="rp-page">Page {page}</span>
                    <span class="rp-meter">
                        <span>Match {similarity:.3f}</span>
                        <span class="rp-meter-track">
                            <span class="rp-meter-fill"
                                  style="display:block;width:{percent:.1f}%">
                            </span>
                        </span>
                    </span>
                </div>
                <div class="rp-muted" style="margin-bottom:8px">
                    Section: {section}
                </div>
                <p class="rp-src-text">{safe_text}</p>
            </div>
            """
        )

    return '<div class="rp-evidence">' + "".join(cards) + "</div>"


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
            "Upload your research PDF and select **Process paper**.",
            EMPTY_SOURCES_HTML,
        )

    if not question or not question.strip():
        return (
            "Please enter a research question.",
            EMPTY_SOURCES_HTML,
        )

    try:
        # Correct argument order: question first, store second
        result = answer_question(
            question.strip(),
            store,
            top_k=3,
        )

        answer = result.get("answer") or "No answer was returned."
        sources = result.get("sources", [])

        if sources:
            source_html = build_sources_html(sources)
        else:
            source_html = (
                '<div class="rp-empty"><div>'
                "No supporting passages were returned. "
                "Try rephrasing your question."
                "</div></div>"
            )

        return answer, source_html

    except Exception as exc:
        return (
            f"**Something went wrong:** {html.escape(str(exc))}",
            EMPTY_SOURCES_HTML,
        )
    
def clear_question():
    return "", ANSWER_PLACEHOLDER, EMPTY_SOURCES_HTML


# ─────────────────────────────────────────────
# User Interface
# ─────────────────────────────────────────────

with gr.Blocks(
    theme=gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="violet",
        neutral_hue="slate",
        radius_size="lg",
        spacing_size="md",
    ),
    css=CUSTOM_CSS,
    title="ResearchPilot AI | Research Assistant",
) as demo:

    gr.HTML(HEADER_HTML)

    document_store = gr.State(value=None)

    with gr.Row(equal_height=True):

        # LEFT: Document workspace
        with gr.Column(scale=4, min_width=280):
            with gr.Group(elem_classes=["rp-card"], elem_id="rp-card-library"):
                gr.HTML(LIBRARY_HTML)

                pdf_input = gr.File(
                    label="Research paper (PDF)",
                    file_types=[".pdf"],
                    type="filepath",
                )

                process_button = gr.Button(
                    "Process paper",
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
            with gr.Group(elem_classes=["rp-card"], elem_id="rp-card-ask"):
                gr.HTML(ASK_HTML)

                question_input = gr.Textbox(
                    label="Your research question",
                    placeholder=(
                        "Example: What methodology does this paper use?"
                    ),
                    lines=3,
                )

                with gr.Row():
                    ask_button = gr.Button(
                        "Get answer",
                        variant="primary",
                        elem_id="rp-ask-btn",
                    )
                    clear_button = gr.Button(
                        "Clear",
                        variant="secondary",
                    )

                gr.HTML('<div class="rp-sub">Answer</div>')

                answer_output = gr.Markdown(
                    value=ANSWER_PLACEHOLDER,
                    elem_id="rp-answer",
                )

    # Evidence area
    with gr.Group(elem_classes=["rp-card"], elem_id="rp-card-evidence"):
        gr.HTML(EVIDENCE_HTML)

        sources_output = gr.HTML(value=EMPTY_SOURCES_HTML)

    gr.HTML(FOOTER_HTML)

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
        fn=clear_question,
        inputs=[],
        outputs=[question_input, answer_output, sources_output],
    )


if __name__ == "__main__":
    demo.launch()