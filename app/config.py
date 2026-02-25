from dataclasses import dataclass
import os

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class Settings:
    postgres_host: str = os.getenv("POSTGRES_HOST", "localhost")
    postgres_port: int = int(os.getenv("POSTGRES_PORT", "5432"))
    postgres_db: str = os.getenv("POSTGRES_DB", "faq_agent")
    postgres_user: str = os.getenv("POSTGRES_USER", "faq_user")
    postgres_password: str = os.getenv("POSTGRES_PASSWORD", "faq_password")
    embedding_model: str = os.getenv("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001")
    chat_model: str = os.getenv("GEMINI_CHAT_MODEL", "gemini-2.5-flash")
    embedding_dimension: int = int(os.getenv("EMBEDDING_DIMENSION", "768"))

    @property
    def postgres_dsn(self) -> str:
        return (
            f"postgresql+psycopg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


settings = Settings()
