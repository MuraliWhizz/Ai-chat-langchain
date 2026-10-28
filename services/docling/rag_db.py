from pathlib import Path

import psycopg


SCHEMA_PATH = Path(__file__).with_name("schema.sql")


def ensure_schema(connection: psycopg.Connection) -> None:
    for statement in SCHEMA_PATH.read_text(encoding="utf-8").split(";"):
        if statement.strip():
            connection.execute(statement)
