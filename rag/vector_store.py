"""Idempotent Gold-to-Qdrant indexing using a named ``text`` vector."""

from __future__ import annotations

from uuid import NAMESPACE_URL, uuid5

from qdrant_client.models import Distance, FieldCondition, Filter, FilterSelector, MatchValue, PointStruct, VectorParams

from rag.build_documents import build_documents, load_gold_tables
from rag.embeddings import get_embedding_model
from rag.settings import QDRANT_COLLECTION, QDRANT_VECTOR_NAME, get_qdrant_client


def ensure_collection(client, vector_size: int) -> None:
    """Create the collection or validate that its named-vector schema matches."""
    from qdrant_client.http.exceptions import UnexpectedResponse

    try:
        info = client.get_collection(QDRANT_COLLECTION)
    except UnexpectedResponse as exc:
        if exc.status_code != 404:
            raise
        info = None
    except ValueError as exc:
        if "collection" not in str(exc).casefold() or "not found" not in str(exc).casefold():
            raise
        info = None

    if info is None:
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config={QDRANT_VECTOR_NAME: VectorParams(size=vector_size, distance=Distance.COSINE)},
        )
        print(f"Created Qdrant collection {QDRANT_COLLECTION!r} ({vector_size}-dimensional named vector).")
        return

    vectors = info.config.params.vectors
    if not isinstance(vectors, dict) or QDRANT_VECTOR_NAME not in vectors:
        raise RuntimeError(
            f"Qdrant collection {QDRANT_COLLECTION!r} uses a legacy unnamed-vector schema. "
            "Set QDRANT_COLLECTION to a new collection name (for example ipl_gold_strategy_v2) "
            "and run the index command again; the existing collection was left untouched."
        )
    configured = vectors[QDRANT_VECTOR_NAME]
    if configured.size != vector_size:
        raise RuntimeError(
            f"Qdrant vector size mismatch: collection has {configured.size}, "
            f"Ollama returned {vector_size}. Use a fresh collection name before reindexing."
        )


def upload_documents(client, embedding_model, documents: list[dict], batch_size: int = 64) -> int:
    """Upsert stable document IDs so rerunning indexing does not duplicate points."""
    if not documents:
        raise ValueError("No documents were generated from the Gold tables.")
    total = len(documents)
    for start in range(0, total, batch_size):
        batch = documents[start:start + batch_size]
        vectors = embedding_model.embed_documents([doc["text"] for doc in batch])
        points = []
        for doc, vector in zip(batch, vectors, strict=True):
            identity = f"{doc['source']}|{doc['key']}"
            points.append(PointStruct(
                id=str(uuid5(NAMESPACE_URL, f"ipl-gold:{identity}")),
                vector={QDRANT_VECTOR_NAME: vector},
                payload={
                    "text": doc["text"],
                    "source": doc["source"],
                    "key": doc["key"],
                    **doc.get("metadata", {}),
                },
            ))
        client.upsert(collection_name=QDRANT_COLLECTION, points=points, wait=True)
        print(f"Indexed {min(start + len(batch), total)}/{total} Gold documents")
    return total


def clear_gold_sources(client, sources: list[str]) -> None:
    """Remove old snapshots for indexed Gold sources before writing replacements."""
    for source in dict.fromkeys(sources):
        client.delete(
            collection_name=QDRANT_COLLECTION,
            points_selector=FilterSelector(
                filter=Filter(must=[FieldCondition(key="source", match=MatchValue(value=source))])
            ),
            wait=True,
        )


def index_gold() -> int:
    tables = load_gold_tables()
    documents = build_documents(tables)
    embedding_model = get_embedding_model()
    sample_vector = embedding_model.embed_query("IPL Gold analytics")
    client = get_qdrant_client()
    ensure_collection(client, len(sample_vector))
    clear_gold_sources(client, ["player_matchups", "player_form", "team_performance", "venue_strategy", "toss_analysis", "phase_analysis"])
    count = upload_documents(client, embedding_model, documents)
    client.close()
    return count


if __name__ == "__main__":
    total = index_gold()
    print(f"Qdrant index ready: {total} searchable Gold documents in {QDRANT_COLLECTION!r}.")
