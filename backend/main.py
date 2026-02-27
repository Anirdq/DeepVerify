"""
DeepVerify – FastAPI Backend (Sprint 1)
Epic 2: Detection Engine

Routes:
  POST /analyze  – Accepts text, runs GLTR + composite scoring, returns results.

Sprint 1 scope: GLTR via GPT-2 (real model) + mock Ghostbuster placeholder.
Ollama/Mistral explanation is mocked for Sprint 1; will be wired in Sprint 2.

Model loading strategy (3-tier):
  1. Try local cache only (local_files_only=True) – instant, no network.
  2. If not cached, try full download – works with internet.
  3. If both fail, run in heuristics-only mode with a clear notice.
Run `python download_model.py` once to pre-cache GPT-2.
"""

import os

# Disable HuggingFace XetHub CDN – falls back to direct HTTP.
# Required on networks where cas-bridge.xethub.hf.co is unreachable.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import logging
import re
import time
from typing import List, Optional

import nltk
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# ─────────────────────────────────────────────────────────
# Logging setup
# ─────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("deepverify")

# ─────────────────────────────────────────────────────────
# Download NLTK sentence tokenizer (first-run only)
# ─────────────────────────────────────────────────────────
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    logger.info("Downloading NLTK punkt_tab tokenizer...")
    nltk.download("punkt_tab", quiet=True)

# ─────────────────────────────────────────────────────────
# Rate limiter (Epic 5, US2)
# ─────────────────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address, default_limits=["10/minute"])

app = FastAPI(
    title="DeepVerify API",
    description="AI-generated text detection backend – Sprint 1",
    version="0.1.0",
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ─────────────────────────────────────────────────────────
# CORS – allow the Vite dev server and any deployed frontend
# ─────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "https://*.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────
# GPT-2 model – 3-tier loading strategy
# ─────────────────────────────────────────────────────────
_tokenizer = None
_model = None
MODEL_AVAILABLE = False  # set to True once a model is ready


def _try_load_model():
    """
    Load GPT-2 from LOCAL CACHE ONLY.
    Network download is intentionally disabled here to prevent startup hangs.
    Run `python download_model.py` separately (with internet) to cache the model.
    """
    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    try:
        tok = GPT2Tokenizer.from_pretrained("gpt2", local_files_only=True)
        mdl = GPT2LMHeadModel.from_pretrained(
            "gpt2", local_files_only=True, use_safetensors=False
        )
        mdl.eval()
        logger.info("GPT-2 loaded from local cache — GLTR mode active.")
        return tok, mdl
    except Exception as e:
        logger.warning(
            f"GPT-2 not in local cache ({type(e).__name__}). "
            "Running in HEURISTICS-ONLY mode. "
            "Run 'python download_model.py' once to enable full GLTR detection."
        )
        return None, None


_tokenizer, _model = _try_load_model()
if _tokenizer is not None:
    MODEL_AVAILABLE = True
    logger.info("Detection engine: GPT-2 GLTR mode active.")
else:
    logger.warning(
        "GPT-2 unavailable — running in HEURISTICS-ONLY mode. "
        "Run 'python download_model.py' once to enable full GLTR detection."
    )


def get_model():
    return _tokenizer, _model


# ─────────────────────────────────────────────────────────
# Pydantic Schemas (Epic 2, US1)
# ─────────────────────────────────────────────────────────
MIN_WORDS = 100
MAX_CHARS = 10_000


class AnalyzeRequest(BaseModel):
    text: str

    @field_validator("text")
    @classmethod
    def validate_text(cls, v: str) -> str:
        v = v.strip()
        if len(v) > MAX_CHARS:
            raise ValueError(f"Text exceeds maximum length of {MAX_CHARS} characters.")
        words = v.split()
        if len(words) < MIN_WORDS:
            raise ValueError(
                f"Text must be at least {MIN_WORDS} words. Got {len(words)}."
            )
        return v


class SentenceScore(BaseModel):
    sentence: str
    risk: int  # 0–100


class AnalyzeResponse(BaseModel):
    score: int
    verdict: str
    explanation: str
    highlighted_text: List[SentenceScore]
    gltr_score: int
    processing_time_ms: int


# ─────────────────────────────────────────────────────────
# GLTR Module (Epic 2, US2) – HuggingFace GPT-2
# ─────────────────────────────────────────────────────────

def compute_gltr_heuristic(text: str) -> float:
    """
    Multi-signal GLTR proxy with HUMAN-SIGNAL suppression.

    AI signals (push score toward 1.0):
      - Sentence-length uniformity
      - Lexical diversity (TTR)
      - AI transition phrases

    Human signals (SUPPRESS the AI score):
      - Informal abbreviations: Plz, msg, ur, gonna, btw, asap ...
      - Typos / alphanumeric mixes: do2wnloaded, III YRs ...
      - ALL-CAPS emphasis (institutional / casual writer habit)
      - Sentences starting with lowercase
      - Standalone lowercase 'i' (texting style)
    """
    import re

    if len(text.split()) < 10:
        return 0.5

    text_lower = text.lower()
    words = text.split()
    words_lower = [w.lower().strip('.,!?;:"') for w in words]

    # ────────────────────────────────────────────────
    # HUMAN SIGNALS
    # ────────────────────────────────────────────────

    # 1. Informal abbreviations / slang (never in AI output)
    slang = [
        'plz', 'pls', 'msg', 'msgs', 'ur', 'u r', 'gonna', 'wanna', 'gotta',
        'btw', 'asap', 'fyi', 'lol', 'omg', 'smh', 'ngl', 'tbh', 'bc',
        'cuz', 'rn', 'imo', 'irl', 'tho', 'thru', 'lemme', 'gimme', 'kinda',
        'sorta', 'dunno', 'ya', 'yep', 'nope', 'yup', 'lmk', 'brb', 'ttyl',
    ]
    slang_hits = sum(1 for s in slang if s in text_lower)
    slang_signal = min(1.0, slang_hits / 1.5)   # even 2 slang = strong human

    # 2. Alphanumeric typo patterns (e.g. 'do2wnloaded', 'III', '7th', '2nd')
    alpha_num = len(re.findall(r'\b(?=[a-zA-Z]*\d)[a-zA-Z\d]{2,}\b', text))
    typo_signal = min(1.0, alpha_num / 3.0)

    # 3. ALL-CAPS words used for emphasis (not standard acronyms)
    common_acronyms = {'ai', 'url', 'api', 'usa', 'gpa', 'hr', 'ceo', 'pdf',
                       'ml', 'ngo', 'sdg', 'phd', 'it', 'iot', 'ok', 'id'}
    caps_words = re.findall(r'\b[A-Z]{2,}\b', text)
    emp_caps = [w for w in caps_words if w.lower() not in common_acronyms]
    caps_signal = min(1.0, len(emp_caps) / 2.0)

    # 4. Sentences / clauses starting with lowercase (texting / informal)
    raw_sents = re.split(r'(?<=[.!?])\s+', text.strip())
    lc_starts = sum(1 for s in raw_sents if s and s[0].islower())
    lc_signal = min(1.0, lc_starts / max(len(raw_sents), 1) * 4.0)

    # 5. Standalone lowercase 'i' as subject
    standalone_i = len(re.findall(r'(?<![a-zA-Z])i(?![a-zA-Z])', text))
    i_signal = min(1.0, standalone_i / 3.0)

    # Combined human boost (any ONE strong signal is enough)
    human_boost = max(slang_signal, typo_signal, caps_signal, lc_signal, i_signal)

    # ────────────────────────────────────────────────
    # AI SIGNALS
    # ────────────────────────────────────────────────

    # 6. Sentence-length uniformity (unreliable below 5 sentences)
    sent_lens = [len(s.split()) for s in raw_sents if s.strip()]
    if len(sent_lens) >= 5:
        mean_sl = sum(sent_lens) / len(sent_lens)
        std_sl  = (sum((l - mean_sl) ** 2 for l in sent_lens) / len(sent_lens)) ** 0.5
        uniformity = max(0.0, 1.0 - std_sl / 12.0)
    else:
        uniformity = 0.2   # too few sentences: don't penalise

    # 7. Lexical diversity TTR
    ttr = len(set(words_lower)) / max(len(words_lower), 1)
    diversity_score = max(0.0, min(1.0, (0.65 - ttr) / 0.30))

    # 8. AI transition/filler phrases
    ai_phrases = [
        'in conclusion', 'furthermore', 'additionally', 'moreover',
        'it is important to note', 'it is worth noting', 'it should be noted',
        'in summary', 'to summarize', 'in addition', 'as a result',
        'it is clear that', 'plays a crucial role', 'plays a vital role',
        'delve into', 'showcasing', 'underscoring', 'tapestry', 'nuanced',
        'it is essential', 'needless to say', 'in terms of', 'consequently',
        'this underscores', 'it is worth mentioning', 'it goes without saying',
    ]
    phrase_hits = sum(1 for p in ai_phrases if p in text_lower)
    phrase_score = min(1.0, phrase_hits / 3.0)

    raw_ai = (
        0.35 * uniformity +
        0.25 * diversity_score +
        0.40 * phrase_score
    )

    # Human signals SUPPRESS the AI score proportionally
    # (human_boost=1.0 cuts the AI score by 80%)
    final = raw_ai * (1.0 - 0.80 * human_boost)
    return float(min(max(final, 0.0), 1.0))


def compute_gltr(text: str, tokenizer, model) -> float:
    """
    Compute a GLTR-style AI score (0.0–1.0) based on token log-probabilities.

    Method:
    1. Tokenise the text with GPT-2.
    2. Compute log-probability of each token given the prior context.
    3. GLTR signal: fraction of tokens whose log-prob rank is in the top-k
       of the full vocabulary (top-10 = very predictable → likely AI).
    4. Also compute burstiness (std dev of log-probs) — low burstiness → AI.
    5. Combine into a single [0,1] score.
    """
    if tokenizer is None or model is None:
        return compute_gltr_heuristic(text)

    import torch
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    input_ids = inputs["input_ids"]

    if input_ids.shape[1] < 5:
        return 0.5  # Too short to be meaningful

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        logits = outputs.logits  # [1, seq_len, vocab_size]

    # Shift: logits[i] predicts token[i+1]
    shift_logits = logits[0, :-1, :]   # [seq_len-1, vocab_size]
    shift_ids = input_ids[0, 1:]       # [seq_len-1]

    log_probs = torch.log_softmax(shift_logits, dim=-1)
    token_log_probs = log_probs[range(len(shift_ids)), shift_ids].detach().numpy()

    # Top-k fraction (k=10): how many tokens are in the top-10 predictions?
    sorted_logits = torch.sort(shift_logits, descending=True, dim=-1).indices
    top_k = 10
    in_top_k = []
    for i, tok_id in enumerate(shift_ids):
        rank = (sorted_logits[i] == tok_id).nonzero(as_tuple=True)[0]
        in_top_k.append(rank.item() < top_k)

    top_k_fraction = sum(in_top_k) / max(len(in_top_k), 1)

    # Burstiness: std dev of log-probs (higher = more human-like variability)
    mean_lp = float(token_log_probs.mean())
    std_lp = float(token_log_probs.std()) if len(token_log_probs) > 1 else 1.0

    # Normalise:
    # top_k_fraction: 0 (human) → 1 (AI)
    # burstiness: high std = human = low score; clamp to [0,1]
    burstiness_score = max(0.0, 1.0 - min(std_lp / 3.0, 1.0))  # invert

    # Weighted blend
    gltr_raw = 0.65 * top_k_fraction + 0.35 * burstiness_score
    return float(min(max(gltr_raw, 0.0), 1.0))


# ─────────────────────────────────────────────────────────
# Ghostbuster placeholder (Epic 2, US3 – Sprint 2 integration)
# ─────────────────────────────────────────────────────────

def compute_ghostbuster_mock(text: str, gltr_score: float) -> float:
    """
    Multi-signal Ghostbuster proxy with context-aware contraction scoring.

    Key fix: contraction ABSENCE is only an AI signal for neutral/standard
    writing register. Institutional memos and informal chat naturally lack
    contractions for different reasons — if we see slang OR emphasis caps,
    we skip the contraction penalty entirely.
    """
    import re

    words = text.split()
    if not words:
        return gltr_score

    text_lower = text.lower()

    # ── Detect writing register ───────────────────────────────
    has_slang = any(s in text_lower for s in
                    ['plz', 'pls', 'msg', 'gonna', 'wanna', 'btw', 'asap', 'rn'])
    has_emp_caps = bool(re.findall(r'\b[A-Z]{3,}\b', text))  # ASSESSMENT, LEVEL
    has_numbers  = bool(re.findall(r'\b\d+\b', text))         # dates, counts
    informal_register = has_slang or has_emp_caps

    # ── 1. Punctuation density ────────────────────────────────
    informal_punct = sum(1 for c in text if c in '!?;()—-…\"\'*')
    punct_per_word = informal_punct / max(len(words), 1)
    punct_score = max(0.0, 1.0 - punct_per_word / 0.10)

    # ── 2. Contraction usage (context-aware) ─────────────────
    contractions = [
        "don't", "can't", "won't", "it's", "i'm", "you're", "they're",
        "we're", "isn't", "aren't", "wasn't", "weren't", "doesn't", "didn't",
        "i've", "you've", "i'll", "you'll", "that's", "there's", "what's",
        "who's", "couldn't", "shouldn't", "wouldn't", "i'd", "we'd",
    ]
    contraction_count = sum(1 for c in contractions if c in text_lower)
    if informal_register:
        # Institutional / very informal text: no penalty for lacking contractions
        contraction_score = 0.25
    else:
        contraction_score = max(0.0, 1.0 - contraction_count / 4.0)

    # ── 3. Paragraph / block uniformity ──────────────────────
    paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
    if len(paragraphs) > 3:
        par_lens = [len(p.split()) for p in paragraphs]
        mean_pl  = sum(par_lens) / len(par_lens)
        std_pl   = (sum((l - mean_pl)**2 for l in par_lens) / len(par_lens)) ** 0.5
        par_uniformity = max(0.0, 1.0 - std_pl / max(mean_pl * 0.9, 1))
    else:
        par_uniformity = 0.3   # too few paragraphs: neutral

    ghost_raw = (
        0.30 * punct_score +
        0.45 * contraction_score +
        0.25 * par_uniformity
    )

    return float(min(max(0.5 * gltr_score + 0.5 * ghost_raw, 0.0), 1.0))


# ─────────────────────────────────────────────────────────
# Composite Scorer (Epic 2, US4)
# ─────────────────────────────────────────────────────────

GLTR_WEIGHT = 0.4
GHOSTBUSTER_WEIGHT = 0.6


def composite_score(gltr: float, ghostbuster: float) -> int:
    """
    Weighted composite: score = (w1*gltr + w2*ghostbuster) / (w1+w2) * 100
    Thresholds: 0-40 human, 41-70 uncertain, 71-100 AI.
    """
    total_w = GLTR_WEIGHT + GHOSTBUSTER_WEIGHT
    raw = (GLTR_WEIGHT * gltr + GHOSTBUSTER_WEIGHT * ghostbuster) / total_w
    return int(round(min(max(raw * 100, 0), 100)))


# ─────────────────────────────────────────────────────────
# Sentence-level scoring (Epic 2, US5 preview)
# ─────────────────────────────────────────────────────────

def score_sentences(
    text: str,
    tokenizer,
    model,
    global_score: int,
) -> List[SentenceScore]:
    """
    Compute per-sentence AI risk scores using NLTK sentence tokenizer
    and per-sentence GLTR. Falls back to proportional variance from global.
    """
    try:
        sentences = nltk.sent_tokenize(text)
    except Exception:
        # Fallback: split on period
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]

    results = []
    for sent in sentences:
        words = sent.split()
        if len(words) < 5:
            # Very short sentence: assign global score with small jitter
            risk = min(100, max(0, global_score + (-8 if len(words) < 3 else 5)))
        else:
            s_gltr = compute_gltr(sent, tokenizer, model)
            s_ghost = compute_ghostbuster_mock(sent, s_gltr)
            risk = composite_score(s_gltr, s_ghost)
        results.append(SentenceScore(sentence=sent, risk=risk))

    return results


# ─────────────────────────────────────────────────────────
# Explanation generator (Sprint 1 local; Ollama in Sprint 2)
# ─────────────────────────────────────────────────────────

def generate_explanation(score: int, verdict: str, highlighted: List[SentenceScore]) -> str:
    """
    Sprint 1 rule-based explanation.
    Sprint 2 will replace this with Ollama/Mistral via:
      POST http://localhost:11434/api/generate  { model: "mistral:7b", prompt: ... }
    """
    high_risk = [s for s in highlighted if s.risk >= 70]
    medium_risk = [s for s in highlighted if 40 <= s.risk < 70]

    if score < 41:
        base = (
            f"This text appears to be written by a human (score: {score}/100). "
            f"The token probability patterns show natural variability consistent with human writing — "
            f"irregular word choices, varied sentence lengths, and lower predictability."
        )
    elif score < 71:
        base = (
            f"This text shows mixed signals (score: {score}/100). "
            f"Some portions read naturally, while others show the uniformity and high token predictability "
            f"characteristic of AI-generated content."
        )
    else:
        base = (
            f"This text is likely AI-generated (score: {score}/100). "
            f"The GLTR analysis detected consistently high token predictability — each word strongly follows "
            f"from its predecessor, a hallmark of large language model outputs."
        )

    if high_risk:
        sample = high_risk[0].sentence[:80] + "…" if len(high_risk[0].sentence) > 80 else high_risk[0].sentence
        base += f' The most suspicious segment: "{sample}"'

    base += " (Note: Sprint 1 uses rule-based explanation. Ollama/Mistral integration planned for Sprint 2.)"
    return base


# ─────────────────────────────────────────────────────────
# API Route (Epic 2, US1)
# ─────────────────────────────────────────────────────────

@app.post("/analyze", response_model=AnalyzeResponse, summary="Analyse text for AI generation")
@limiter.limit("10/minute")
async def analyze(request: Request, body: AnalyzeRequest) -> AnalyzeResponse:
    """
    POST /analyze

    Accepts text, runs GLTR + composite scoring, returns:
    - score (0-100)
    - verdict ("Likely Human" | "Uncertain" | "Likely AI")
    - explanation (plain English)
    - highlighted_text (per-sentence risk scores)
    - gltr_score (0-100)
    - processing_time_ms
    """
    start = time.time()
    text = body.text
    logger.info(f"Analyzing text: {len(text)} chars, {len(text.split())} words")

    tokenizer, model = get_model()

    # Run detection
    gltr_raw = compute_gltr(text, tokenizer, model)
    ghost_raw = compute_ghostbuster_mock(text, gltr_raw)
    final_score = composite_score(gltr_raw, ghost_raw)

    # Verdict
    if final_score < 41:
        verdict = "Likely Human"
    elif final_score < 71:
        verdict = "Uncertain"
    else:
        verdict = "Likely AI"

    # Sentence scoring
    highlighted = score_sentences(text, tokenizer, model, final_score)

    # Explanation
    explanation = generate_explanation(final_score, verdict, highlighted)

    elapsed_ms = int((time.time() - start) * 1000)
    logger.info(f"Result: score={final_score}, verdict={verdict}, time={elapsed_ms}ms")

    return AnalyzeResponse(
        score=final_score,
        verdict=verdict,
        explanation=explanation,
        highlighted_text=highlighted,
        gltr_score=int(round(gltr_raw * 100)),
        processing_time_ms=elapsed_ms,
    )


@app.get("/health", summary="Health check")
async def health():
    return {"status": "ok", "sprint": 1}
