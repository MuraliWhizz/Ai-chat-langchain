import argparse
import hashlib
import json
from pathlib import Path

import psycopg
from docling.document_converter import DocumentConverter

from parser import SUPPORTED_EXTENSIONS, parse_document
from rag_db import ensure_schema
from rag_utils import (
    embed_texts,
    required_setting,
    vector_literal,
)


CHUNK_SIZE = 1200
CHUNK_OVERLAP = 150


def chunk_text(
    text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP
) -> list[str]:
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Require chunk_size > overlap >= 0")

    text = text.strip()
    if not text:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            boundary = text.rfind("\n", start + chunk_size // 2, end)
            if boundary > start:
                end = boundary

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break

        start = max(start + 1, end - overlap)
        while start < len(text) and text[start].isspace():
            start += 1

    return chunks


def discover_files(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Input path does not exist: {path}")
        candidates = path.rglob("*") if path.is_dir() else [path]
        for candidate in candidates:
            if candidate.is_file() and candidate.suffix.lower() in SUPPORTED_EXTENSIONS:
                files.append(candidate.resolve())
    unique_files = sorted(set(files))
    if not unique_files:
        raise ValueError(
            "No supported documents found. Supported extensions: "
            + ", ".join(sorted(SUPPORTED_EXTENSIONS))
        )
    return unique_files


def ingest_file(
    connection: psycopg.Connection,
    source: Path,
    converter: DocumentConverter,
) -> dict[str, str | int]:
    content_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    existing = connection.execute(
        "SELECT id, sha256 FROM rag_documents WHERE source_path = %s",
        (str(source),),
    ).fetchone()
    if existing and existing[1].strip() == content_hash:
        return {"source": str(source), "status": "unchanged", "chunks": 0}

    markdown = parse_document(source, converter)
    chunks = chunk_text(markdown)
    if not chunks:
        raise ValueError(f"No text chunks were produced for {source}")

    embeddings = embed_texts(chunks, prefix="search_document: ")

    if len(embeddings) != len(chunks):
        raise RuntimeError(f"Embedding count did not match chunk count for {source}")

    document = connection.execute(
        """
        INSERT INTO rag_documents (source_path, sha256)
        VALUES (%s, %s)
        ON CONFLICT (source_path) DO UPDATE
        SET sha256 = EXCLUDED.sha256, updated_at = now()
        RETURNING id
        """,
        (str(source), content_hash),
    ).fetchone()
    if document is None:
        raise RuntimeError(f"Could not save document metadata for {source}")
    document_id = document[0]
    connection.execute("DELETE FROM rag_chunks WHERE document_id = %s", (document_id,))
    with connection.cursor() as cursor:
        cursor.executemany(
            """
            INSERT INTO rag_chunks (document_id, chunk_index, content, embedding)
            VALUES (%s, %s, %s, %s::vector)
            """,
            [
                (document_id, index, chunk, vector_literal(embedding))
                for index, (chunk, embedding) in enumerate(
                    zip(chunks, embeddings, strict=True)
                )
            ],
        )
    return {"source": str(source), "status": "indexed", "chunks": len(chunks)}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse local documents, embed them with Ollama, and store them in pgvector."
    )
    parser.add_argument(
        "paths", nargs="+", type=Path, help="Files or directories to index"
    )
    parser.add_argument(
        "--json", action="store_true", help="Print results as a JSON array"
    )
    args = parser.parse_args()
    files = discover_files(args.paths)

    with psycopg.connect(required_setting("DATABASE_URL")) as connection:
        ensure_schema(connection)
        converter = DocumentConverter()
        results = [ingest_file(connection, source, converter) for source in files]

    if args.json:
        print(json.dumps(results))
    else:
        for result in results:
            if result is None:
                continue
            if result["status"] == "unchanged":
                print(f"Unchanged, skipped: {result['source']}")
            else:
                print(f"Indexed {result['source']}: {result['chunks']} chunks")


if __name__ == "__main__":
    main()