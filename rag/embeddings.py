"""Ollama embedding model factory."""

from langchain_ollama import OllamaEmbeddings

from rag.settings import OLLAMA_BASE_URL, OLLAMA_EMBED_MODEL


def get_embedding_model() -> OllamaEmbeddings:
    return OllamaEmbeddings(model=OLLAMA_EMBED_MODEL, base_url=OLLAMA_BASE_URL)


if __name__ == "__main__":
    vector = get_embedding_model().embed_query("IPL venue chasing record")
    print(f"Embedding generated: {len(vector)} dimensions")
