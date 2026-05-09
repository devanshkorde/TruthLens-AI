"""
TruthLens AI — XAI Explainer Utility
======================================
Provides enhanced human-readable explanations for model predictions.
Optionally uses Google Gemini API for richer explanation text.
"""

import os
import re

# Optional Gemini enhancement
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

MANIPULATION_TACTICS = {
    "appeal to fear": [
        r"danger", r"threat", r"attack", r"crisis", r"chaos", r"catastrophe",
        r"collapse", r"destroy", r"end of", r"war", r"invasion",
    ],
    "false urgency": [
        r"right now", r"breaking", r"alert", r"urgent", r"emergency",
        r"immediately", r"act now", r"before it'?s too late",
    ],
    "ad hominem": [
        r"stupid", r"idiot", r"moron", r"corrupt", r"evil", r"criminal",
        r"traitor", r"liar", r"fraud",
    ],
    "conspiracy framing": [
        r"they don'?t want", r"hidden agenda", r"cover.?up", r"deep state",
        r"new world order", r"conspiracy", r"controlled", r"puppet",
    ],
    "fake statistics": [
        r"\d+%? (of|say|believe|think|claim)", r"studies show", r"experts say",
        r"scientists confirm",  # without citing who
    ],
}


def detect_manipulation_tactics(text: str) -> list[str]:
    """Return list of detected psychological manipulation tactics."""
    found = []
    text_lower = text.lower()
    for tactic, patterns in MANIPULATION_TACTICS.items():
        for pattern in patterns:
            if re.search(pattern, text_lower):
                found.append(tactic)
                break
    return found


def check_hinglish(text: str) -> bool:
    """Detect if text contains Hinglish or Hindi transliterated content."""
    hinglish_markers = [
        "hai", "nahi", "kya", "yeh", "woh", "aur", "ka", "ki", "ke",
        "mein", "ko", "ne", "se", "par", "bhi", "toh", "lekin", "agar",
        "nakli", "jhooth", "sach", "share", "viral", "desh", "sarkar",
    ]
    text_lower = text.lower()
    count = sum(1 for marker in hinglish_markers if f" {marker} " in text_lower)
    return count >= 3


def generate_explanation(
    text: str,
    verdict: str,
    confidence: float,
    signals: list[str],
) -> dict:
    """
    Generate a comprehensive explanation for the verdict.
    Uses Gemini API if available, otherwise builds structured explanation locally.
    """
    tactics = detect_manipulation_tactics(text)
    is_hinglish = check_hinglish(text)

    explanation = {
        "summary": _build_summary(verdict, confidence),
        "manipulation_tactics": tactics,
        "is_hinglish": is_hinglish,
        "language_note": "Hinglish/Hindi text detected — regional context applied" if is_hinglish else None,
        "detailed_signals": signals,
        "verdict_explanation": _verdict_explanation(verdict, confidence),
        "recommendation": _build_recommendation(verdict),
    }

    # Optionally enhance with Gemini
    if GEMINI_API_KEY and len(text) > 50:
        try:
            import google.generativeai as genai
            genai.configure(api_key=GEMINI_API_KEY)
            model = genai.GenerativeModel("gemini-1.5-flash")

            prompt = f"""You are a media literacy expert. A user submitted this text for fake news analysis.
Our AI model predicted: {verdict} (confidence: {confidence*100:.0f}%).

Text (first 500 chars): {text[:500]}

In 2-3 sentences, explain WHY this might be {verdict} from a journalistic perspective. 
Focus on: writing style, emotional tone, factual grounding. Be objective and educational.
Do NOT provide your own verdict — just explain the signals."""

            response = model.generate_content(prompt)
            explanation["gemini_insight"] = response.text.strip()
        except Exception as e:
            explanation["gemini_insight"] = None

    return explanation


def _build_summary(verdict: str, confidence: float) -> str:
    pct = confidence * 100
    if verdict == "FAKE":
        return f"This content shows strong indicators of misinformation (confidence: {pct:.0f}%)"
    elif verdict == "SUSPICIOUS":
        return f"This content has mixed signals — proceed with caution (confidence: {pct:.0f}%)"
    else:
        return f"This content appears credible based on linguistic analysis (confidence: {pct:.0f}%)"


def _verdict_explanation(verdict: str, confidence: float) -> str:
    if verdict == "FAKE":
        return (
            "The text exhibits multiple hallmarks of misinformation: emotionally charged language, "
            "absence of verifiable sources, and patterns commonly found in misleading content."
        )
    elif verdict == "SUSPICIOUS":
        return (
            "The content has some credibility signals but also contains elements common in "
            "biased or poorly-sourced reporting. Independent verification is recommended."
        )
    else:
        return (
            "The text follows journalistic conventions: measured language, source citations, "
            "and factual framing consistent with credible reporting."
        )


def _build_recommendation(verdict: str) -> str:
    if verdict == "FAKE":
        return "❌ Do not share. Cross-check with Reuters, AP, BBC, or regional fact-checkers like BoomLive or AltNews."
    elif verdict == "SUSPICIOUS":
        return "⚠️ Verify before sharing. Check original source and look for corroboration from 2+ trusted outlets."
    else:
        return "✅ Appears reliable. Always verify major claims through primary sources."
