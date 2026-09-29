"""Similarity retrieval over the Gold text documents in Qdrant."""

from __future__ import annotations

from qdrant_client.models import FieldCondition, Filter, MatchValue

from rag.embeddings import get_embedding_model
from rag.settings import QDRANT_COLLECTION, QDRANT_VECTOR_NAME, get_qdrant_client


class GoldRAGRetriever:
    def __init__(self, client=None, embedding_model=None) -> None:
        self.client = client or get_qdrant_client()
        self.embedding_model = embedding_model or get_embedding_model()

    def search(self, query: str, top_k: int = 5, sources: list[str] | None = None) -> list[dict]:
        if not query.strip():
            return []
        query_vector = self.embedding_model.embed_query(query)
        query_filter = None
        if sources:
            query_filter = Filter(should=[
                FieldCondition(key="source", match=MatchValue(value=source))
                for source in sources
            ])
        result = self.client.query_points(
            collection_name=QDRANT_COLLECTION,
            query=query_vector,
            using=QDRANT_VECTOR_NAME,
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
        )
        return [
            {
                "text": point.payload.get("text", ""),
                "source": point.payload.get("source", "unknown"),
                "key": point.payload.get("key", ""),
                "score": point.score,
            }
            for point in result.points
        ]

    def close(self) -> None:
        self.client.close()
