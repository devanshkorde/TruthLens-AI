"""
TruthLens AI — Text Fake News Model
====================================
Uses TF-IDF + Logistic Regression trained on a synthetic + curated dataset.
Falls back to heuristic scoring when model file is absent (demo mode).

Run `python train_model.py` once to generate the .pkl files.
"""

import os
import re
import json
import math
import joblib
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field

MODEL_DIR = Path(__file__).parent.parent / "data"
VECTORIZER_PATH = MODEL_DIR / "tfidf_vectorizer.pkl"
CLASSIFIER_PATH = MODEL_DIR / "text_classifier.pkl"

# ──────────────────────────────────────────────
# Linguistic feature constants
# ──────────────────────────────────────────────
EMOTIONAL_WORDS = {
    "shocking", "unbelievable", "outrageous", "breaking", "exclusive",
    "exposed", "revealed", "secret", "conspiracy", "hoax", "alert",
    "urgent", "viral", "bombshell", "scandalous", "terrifying", "devastating",
    "horrifying", "sensational", "jaw-dropping", "explosive", "massive",
    # Hinglish / Hindi-transliterated emotional words
    "nakli", "jhooth", "propaganda", "andolan", "danga", "tukde",
}

CREDIBILITY_SIGNALS = {
    "according to", "sources say", "study shows", "report finds",
    "researchers", "officials", "government", "data shows", "statistics",
    "published in", "peer-reviewed", "survey", "evidence",
}

CLICKBAIT_PATTERNS = [
    r"you won't believe",
    r"what (they|he|she|the government) (don't|doesn't|won'?t) want you to know",
    r"\d+ (reasons|things|ways|facts|secrets)",
    r"(this|watch) will (shock|amaze|surprise) you",
    r"share (this|before it'?s deleted)",
    r"(doctors|scientists|experts) (hate|don'?t want you|are hiding)",
]


@dataclass
class TextAnalysisResult:
    verdict: str
    confidence: float
    signals: list[str] = field(default_factory=list)
    highlighted_phrases: list[str] = field(default_factory=list)
    sentiment_score: float = 0.0
    credibility_signals_found: int = 0
    emotional_word_density: float = 0.0


# ──────────────────────────────────────────────
# Feature extraction
# ──────────────────────────────────────────────

def extract_features(text: str) -> dict:
    text_lower = text.lower()
    words = re.findall(r'\b\w+\b', text_lower)
    word_count = max(len(words), 1)
    sentences = re.split(r'[.!?]+', text)
    sentence_count = max(len([s for s in sentences if s.strip()]), 1)

    # Emotional word density
    emotional_hits = [w for w in words if w in EMOTIONAL_WORDS]
    emotional_density = len(emotional_hits) / word_count

    # Credibility signal count
    credibility_hits = sum(1 for sig in CREDIBILITY_SIGNALS if sig in text_lower)

    # Caps ratio (SCREAMING indicates sensationalism)
    alpha_chars = [c for c in text if c.isalpha()]
    caps_ratio = sum(1 for c in alpha_chars if c.isupper()) / max(len(alpha_chars), 1)

    # Exclamation density
    exclamation_density = text.count("!") / word_count

    # Question mark density
    question_density = text.count("?") / word_count

    # Clickbait pattern matches
    clickbait_hits = sum(1 for p in CLICKBAIT_PATTERNS if re.search(p, text_lower))

    # Average word length (shorter = more tabloid-like)
    avg_word_len = np.mean([len(w) for w in words]) if words else 5.0

    # Avg sentence length (extremely short = headline bait)
    avg_sentence_len = word_count / sentence_count

    return {
        "emotional_density": emotional_density,
        "credibility_signals": credibility_hits,
        "caps_ratio": caps_ratio,
        "exclamation_density": exclamation_density,
        "question_density": question_density,
        "clickbait_hits": clickbait_hits,
        "avg_word_len": avg_word_len,
        "avg_sentence_len": avg_sentence_len,
        "word_count": word_count,
    }


def get_highlighted_phrases(text: str) -> list[str]:
    """Return suspicious phrases found in the text."""
    highlights = []
    text_lower = text.lower()

    for word in EMOTIONAL_WORDS:
        if word in text_lower:
            highlights.append(word)

    for pattern in CLICKBAIT_PATTERNS:
        match = re.search(pattern, text_lower)
        if match:
            highlights.append(match.group(0)[:60])

    return list(set(highlights))[:8]


def build_signals(features: dict, verdict: str) -> list[str]:
    """Generate human-readable XAI explanation signals."""
    signals = []

    if features["emotional_density"] > 0.05:
        signals.append("🔴 High emotional word density detected — typical of sensationalist content")
    if features["caps_ratio"] > 0.15:
        signals.append("🔴 Excessive use of CAPITAL LETTERS — common in misleading headlines")
    if features["exclamation_density"] > 0.02:
        signals.append("🟡 Overuse of exclamation marks — indicates emotional manipulation")
    if features["credibility_signals"] == 0:
        signals.append("🔴 No verifiable sources or citations found in text")
    elif features["credibility_signals"] >= 3:
        signals.append("🟢 Multiple credibility signals detected (sources, data references)")
    if features["clickbait_hits"] > 0:
        signals.append(f"🔴 Clickbait pattern detected — {features['clickbait_hits']} trigger phrase(s) found")
    if features["avg_sentence_len"] < 8:
        signals.append("🟡 Very short sentences — typical of headline-bait style writing")
    if features["avg_word_len"] > 6 and features["credibility_signals"] > 1:
        signals.append("🟢 Academic/formal vocabulary with source citations — indicates credibility")
    if features["question_density"] > 0.03:
        signals.append("🟡 Heavy use of rhetorical questions — common persuasion tactic")

    if not signals:
        if verdict == "REAL":
            signals.append("🟢 Writing style consistent with credible journalism")
        else:
            signals.append("🟡 Insufficient signals for definitive classification")

    return signals


# ──────────────────────────────────────────────
# Heuristic scoring (always available as fallback)
# ──────────────────────────────────────────────

def heuristic_score(features: dict) -> tuple[str, float]:
    """
    Score-based fake news heuristic.
    Returns (verdict, confidence_0_to_1).
    """
    fake_score = 0.0

    fake_score += features["emotional_density"] * 5.0
    fake_score += features["caps_ratio"] * 3.0
    fake_score += features["exclamation_density"] * 4.0
    fake_score += features["clickbait_hits"] * 1.5
    fake_score -= features["credibility_signals"] * 0.8
    if features["avg_sentence_len"] < 8:
        fake_score += 0.5
    if features["avg_word_len"] < 4:
        fake_score += 0.3

    # Normalize to 0–1
    normalized = min(max(fake_score / 6.0, 0.0), 1.0)

    if normalized > 0.65:
        return "FAKE", normalized
    elif normalized > 0.35:
        return "SUSPICIOUS", 0.5 + abs(normalized - 0.5)
    else:
        return "REAL", 1.0 - normalized


# ──────────────────────────────────────────────
# Main analysis function
# ──────────────────────────────────────────────

def analyze_text(text: str) -> TextAnalysisResult:
    features = extract_features(text)
    highlighted = get_highlighted_phrases(text)

    verdict = "REAL"
    confidence = 0.85

    # Try ML model first
    if VECTORIZER_PATH.exists() and CLASSIFIER_PATH.exists():
        try:
            vectorizer = joblib.load(VECTORIZER_PATH)
            classifier = joblib.load(CLASSIFIER_PATH)
            vec = vectorizer.transform([text])
            pred = classifier.predict(vec)[0]
            proba = classifier.predict_proba(vec)[0]
            verdict = "FAKE" if pred == 1 else "REAL"
            confidence = float(max(proba))

            # Override with SUSPICIOUS if confidence is low
            if confidence < 0.65:
                verdict = "SUSPICIOUS"
        except Exception as e:
            print(f"[TextModel] ML model error, using heuristic: {e}")
            verdict, confidence = heuristic_score(features)
    else:
        verdict, confidence = heuristic_score(features)

    signals = build_signals(features, verdict)

    return TextAnalysisResult(
        verdict=verdict,
        confidence=round(confidence, 3),
        signals=signals,
        highlighted_phrases=highlighted,
        sentiment_score=round(features["emotional_density"], 3),
        credibility_signals_found=features["credibility_signals"],
        emotional_word_density=round(features["emotional_density"], 3),
    )
