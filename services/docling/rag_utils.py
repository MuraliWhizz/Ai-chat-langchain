import math
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings


ROOT = Path(__file__).resolve().parents[2]
EMBEDDING_DIMENSIONS = 1024
EMBED_BATCH_SIZE = 16

load_dotenv(ROOT / ".env")
load_dotenv(ROOT / "apps" / "api" / ".env")


def required_setting(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(
            f"{name} is not configured. Set it in the project .env file."
        )
    return value


def ollama_base_url() -> str:
    return os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")


def embedding_model() -> str:
    return os.getenv("OLLAMA_EMBED_MODEL", "qwen3-embedding:0.6b")


def chat_model() -> str:
    return os.getenv("OLLAMA_CHAT_MODEL", "qwen3:4b")


def ollama_embeddings() -> OllamaEmbeddings:
    return OllamaEmbeddings(
        model=embedding_model(),
        base_url=ollama_base_url(),
        keep_alive=300,
        sync_client_kwargs={"timeout": 180},
    )


def vector_literal(values: list[float]) -> str:
    if len(values) != EMBEDDING_DIMENSIONS:
        raise ValueError(
            f"Expected {EMBEDDING_DIMENSIONS} embedding dimensions, got {len(values)}"
        )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Embedding contains a non-finite value")
    return "[" + ",".join(str(value) for value in values) + "]"


def embed_texts(texts: list[str], prefix: str) -> list[list[float]]:
    if not texts:
        return []

    embeddings: list[list[float]] = []
    model = ollama_embeddings()
    for offset in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = [
            f"{prefix}{text}"
            for text in texts[offset : offset + EMBED_BATCH_SIZE]
        ]
        batch_embeddings = model.embed_documents(batch)
        if len(batch_embeddings) != len(batch):
            raise RuntimeError("Ollama returned an unexpected embedding count")
        for embedding in batch_embeddings:
            numeric_embedding = [float(value) for value in embedding]
            vector_literal(numeric_embedding)
            embeddings.append(numeric_embedding)

    return embeddings
