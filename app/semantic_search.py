from app.llm_clients import build_embeddings_client
from app.vector_store import ensure_vector_schema, similarity_search
from app.config import settings


def search_relevant_chunks(question: str, top_k: int = 4) -> list[dict]:
    ensure_vector_schema()
    embeddings = build_embeddings_client()
    query_vector = embeddings.embed_query(
        question, output_dimensionality=settings.embedding_dimension
    )
    return similarity_search(query_vector, top_k=top_k)

