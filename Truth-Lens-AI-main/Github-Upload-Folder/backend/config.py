"""
TruthLens AI — Central Configuration
======================================
Loads all settings from .env file.
All other modules import from here — never import os.getenv directly.

Providers (in priority order):
  1. NVIDIA NIM — llama-3.2-nemoretriever-500m-rerank-v2  (primary reranker)
  2. Gemini      — gemini-1.5-flash                        (generative fallback)
  3. Groq        — llama-3.3-70b-versatile                 (ultra-fast fallback)
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from backend directory
ENV_PATH = Path(__file__).parent / ".env"
load_dotenv(ENV_PATH)


class AIConfig:
    """AI provider settings."""

    # ── NVIDIA NIM (Primary — Reranker-based fact verification) ───────────────
    NVIDIA_API_KEY: str = os.getenv("NVIDIA_API_KEY", "")

    # ── Other AI providers (Generative fallbacks) ─────────────────────────────
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")

    # "nvidia" | "gemini" | "groq" | "auto"
    PROVIDER: str = os.getenv("AI_PROVIDER", "auto")
    MAX_TOKENS: int = int(os.getenv("AI_MAX_TOKENS", "300"))
    ENABLED: bool = os.getenv("AI_FACT_CHECK_ENABLED", "true").lower() == "true"

    @classmethod
    def has_nvidia(cls) -> bool:
        return bool(cls.NVIDIA_API_KEY and cls.NVIDIA_API_KEY not in ("", "your_nvidia_api_key_here"))

    @classmethod
    def has_gemini(cls) -> bool:
        return bool(cls.GEMINI_API_KEY and cls.GEMINI_API_KEY != "your_gemini_api_key_here")

    @classmethod
    def has_groq(cls) -> bool:
        return bool(cls.GROQ_API_KEY and cls.GROQ_API_KEY != "your_groq_api_key_here")

    @classmethod
    def active_provider(cls) -> str | None:
        """Return which provider is actually configured (priority: nvidia > gemini > groq)."""
        if cls.PROVIDER == "auto":
            if cls.has_nvidia():   return "nvidia"
            if cls.has_gemini():   return "gemini"
            if cls.has_groq():     return "groq"
            return None
        if cls.PROVIDER == "nvidia"  and cls.has_nvidia():   return "nvidia"
        if cls.PROVIDER == "gemini"  and cls.has_gemini():   return "gemini"
        if cls.PROVIDER == "groq"    and cls.has_groq():     return "groq"
        return None


class ServerConfig:
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    DEBUG: bool = os.getenv("DEBUG", "true").lower() == "true"


# Singleton instances
ai_config = AIConfig()
server_config = ServerConfig()
