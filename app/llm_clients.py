import os

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from app.config import settings


def _require_gemini_key() -> None:
    key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not key or not key.strip():
        raise ValueError(
            "Gemini API key is required. Set GOOGLE_API_KEY or GEMINI_API_KEY in your .env file."
        )


def build_embeddings_client() -> GoogleGenerativeAIEmbeddings:
    _require_gemini_key()
    return GoogleGenerativeAIEmbeddings(model=settings.embedding_model)


def build_chat_client(temperature: float = 0.0) -> ChatGoogleGenerativeAI:
    _require_gemini_key()
    return ChatGoogleGenerativeAI(
        model=settings.chat_model,
        temperature=temperature,
        timeout=120,
    )

