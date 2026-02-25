import logging
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

logger = logging.getLogger(__name__)


def extract_pdf_text(path: Path) -> str:
    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages).strip()


def split_text(text: str) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return [chunk.strip() for chunk in splitter.split_text(text) if chunk.strip()]


def load_pdf_chunks(folder_path: Path) -> list[tuple[str, int, str]]:
    all_chunks: list[tuple[str, int, str]] = []
    pdf_paths = sorted(folder_path.glob("*.pdf"))
    logger.info("Found %d PDF(s) in %s", len(pdf_paths), folder_path)

    for pdf_path in pdf_paths:
        logger.info("Processing %s", pdf_path.name)
        text = extract_pdf_text(pdf_path)
        chunks = split_text(text)
        for idx, chunk in enumerate(chunks):
            all_chunks.append((pdf_path.name, idx, chunk))
        logger.info("  %s: %d chunks", pdf_path.name, len(chunks))
    return all_chunks

