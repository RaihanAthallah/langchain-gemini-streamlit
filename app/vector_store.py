from typing import Any

from pgvector.psycopg import register_vector
from psycopg.rows import dict_row

from app.config import settings
from app.db import get_connection


def ensure_vector_schema() -> None:
    with get_connection() as connection:
        connection.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        connection.execute(
            f"""
            CREATE TABLE IF NOT EXISTS faq_chunks (
                id BIGSERIAL PRIMARY KEY,
                source_file TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                content TEXT NOT NULL,
                embedding VECTOR({settings.embedding_dimension}) NOT NULL
            );
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS faq_chunks_embedding_idx
            ON faq_chunks
            USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100);
            """
        )
        connection.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS faq_chunks_source_idx
            ON faq_chunks (source_file, chunk_index);
            """
        )
        connection.commit()


def upsert_chunk(
    source_file: str,
    chunk_index: int,
    content: str,
    embedding: list[float],
) -> None:
    with get_connection() as connection:
        register_vector(connection)
        connection.execute(
            """
            INSERT INTO faq_chunks (source_file, chunk_index, content, embedding)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (source_file, chunk_index)
            DO UPDATE SET content = EXCLUDED.content, embedding = EXCLUDED.embedding;
            """,
            (source_file, chunk_index, content, embedding),
        )
        connection.commit()


def similarity_search(
    query_embedding: list[float],
    top_k: int = 4,
) -> list[dict[str, Any]]:
    with get_connection() as connection:
        register_vector(connection)
        connection.row_factory = dict_row
        rows = connection.execute(
            """
            SELECT
                id,
                source_file,
                chunk_index,
                content,
                1 - (embedding <=> %s::vector) AS score
            FROM faq_chunks
            ORDER BY embedding <=> %s::vector
            LIMIT %s;
            """,
            (query_embedding, query_embedding, top_k),
        ).fetchall()
        return list(rows)

