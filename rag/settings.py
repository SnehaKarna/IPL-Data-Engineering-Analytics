"""Shared local Ollama and Qdrant configuration."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "mistral")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
QDRANT_URL = os.getenv("QDRANT_URL", "")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") or None
QDRANT_PATH = os.getenv("QDRANT_PATH", "")
QDRANT_BACKEND = os.getenv("QDRANT_BACKEND", "cloud" if QDRANT_URL else "local" if QDRANT_PATH else "")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "ipl_gold_strategy_v2")
QDRANT_VECTOR_NAME = "text"


def get_qdrant_client():
    """Create a cloud/server client or an embedded local client."""
    from qdrant_client import QdrantClient

    if QDRANT_BACKEND == "local" and QDRANT_PATH:
        path = Path(QDRANT_PATH)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        path.mkdir(parents=True, exist_ok=True)
        return QdrantClient(path=str(path))
    if QDRANT_BACKEND == "cloud" and QDRANT_URL:
        return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60)
    raise RuntimeError(
        "Configure QDRANT_BACKEND=local with QDRANT_PATH, or QDRANT_BACKEND=cloud with QDRANT_URL, in .env."
    )
