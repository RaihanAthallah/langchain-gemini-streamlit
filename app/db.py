from contextlib import contextmanager
from typing import Iterator

import psycopg

from app.config import settings


CONNECT_TIMEOUT = 5


def psycopg_dsn() -> str:
    return (
        f"host={settings.postgres_host} "
        f"port={settings.postgres_port} "
        f"dbname={settings.postgres_db} "
        f"user={settings.postgres_user} "
        f"password={settings.postgres_password} "
        f"connect_timeout={CONNECT_TIMEOUT}"
    )


@contextmanager
def get_connection() -> Iterator[psycopg.Connection]:
    connection = psycopg.connect(psycopg_dsn())
    try:
        yield connection
    finally:
        connection.close()

