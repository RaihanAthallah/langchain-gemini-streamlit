import logging
from pathlib import Path

from app.config import settings
from app.document_processing import load_pdf_chunks
from app.llm_clients import build_embeddings_client
from app.vector_store import ensure_vector_schema, upsert_chunk

logger = logging.getLogger(__name__)


def ingest_pdf_folder(folder_path: Path) -> int:
    logger.info("Starting ingestion from %s", folder_path)
    logger.info("Connecting to PostgreSQL...")
    ensure_vector_schema()
    logger.info("Vector schema ensured")

    chunks = load_pdf_chunks(folder_path)
    if not chunks:
        logger.warning("No PDF chunks found in %s", folder_path)
        return 0

    sources = {c[0] for c in chunks}
    logger.info("Loaded %d chunks from %d file(s): %s", len(chunks), len(sources), sorted(sources))

    logger.info("Building embeddings (dim=%d)...", settings.embedding_dimension)
    embeddings = build_embeddings_client()
    texts = [content for _, _, content in chunks]
    vectors = embeddings.embed_documents(
        texts, output_dimensionality=settings.embedding_dimension
    )
    logger.info("Embedded %d chunks", len(vectors))

    for i, ((source_file, chunk_index, content), vector) in enumerate(zip(chunks, vectors)):
        upsert_chunk(source_file, chunk_index, content, vector)
        if (i + 1) % 50 == 0 or i + 1 == len(chunks):
            logger.info("Upserted %d/%d chunks", i + 1, len(chunks))

    logger.info("Ingestion complete. Total chunks: %d", len(chunks))
    return len(chunks)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    base_knowledge = Path("base-knowledge")
    count = ingest_pdf_folder(base_knowledge)
    print(f"Ingested {count} chunks from {base_knowledge}.")

