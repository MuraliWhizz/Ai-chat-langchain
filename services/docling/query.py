import argparse
import json
import os
import re
from pathlib import Path

import psycopg
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_ollama import ChatOllama

from rag_db import ensure_schema
from rag_utils import (
    chat_model,
    ollama_base_url,
    ollama_embeddings,
    required_setting,
    vector_literal,
)


QUESTION_STOP_WORDS = {
    "about",
    "according",
    "and",
    "are",
    "based",
    "background",
    "brief",
    "can",
    "describe",
    "development",
    "explain",
    "experience",
    "for",
    "give",
    "full",
    "he",
    "him",
    "his",
    "how",
    "is",
    "me",
    "more",
    "of",
    "on",
    "please",
    "resume",
    "source",
    "stack",
    "suitable",
    "tell",
    "some",
    "the",
    "this",
    "web",
    "what",
    "who",
    "why",
}


def question_mentions_prior_source(
    question: str, history: list[dict[str, object]]
) -> bool:
    question_terms = set(re.findall(r"[a-zA-Z0-9]+", question.lower()))
    for turn in history:
        sources = turn.get("sources")
        if turn.get("role") != "assistant" or not isinstance(sources, list):
            continue
        for source in sources:
            if not isinstance(source, dict) or not isinstance(source.get("source"), str):
                continue
            basename = os.path.basename(source["source"]).rsplit(".", 1)[0]
            source_terms = {
                term
                for term in re.findall(r"[a-zA-Z0-9]+", basename.lower())
                if len(term) > 3 and term not in QUESTION_STOP_WORDS
            }
            if source_terms & question_terms:
                return True
    return False


def format_history(
    history: list[dict[str, object]], question: str = ""
) -> str:
    if question and question_mentions_prior_source(question, history):
        return ""
    user_turns = [
        turn
        for turn in history
        if turn.get("role") == "user"
        and isinstance(turn.get("content"), str)
        and len(turn["content"]) <= 500
    ]
    if not user_turns:
        return ""
    return f"user: {user_turns[-1]['content']}"


def answer_style(question: str) -> str:
    if re.search(r"\b(brief|briefly|short|quick)\b", question, re.IGNORECASE):
        return (
            "The user asked for a brief follow-up: answer in 3-5 concise sentences, "
            "add useful resume details, and do not repeat the prior comparison."
        )
    return "Answer in 2 concise paragraphs, up to 180 words."


def referenced_source(question: str, history: list[dict[str, object]]) -> str | None:
    match = re.search(r"\bsource\s+(\d+)\b", question, re.IGNORECASE)
    if not match:
        return None
    source_number = int(match.group(1))
    for turn in reversed(history):
        sources = turn.get("sources")
        if turn.get("role") != "assistant" or not isinstance(sources, list):
            continue
        if not 1 <= source_number <= len(sources):
            return None
        source = sources[source_number - 1]
        if isinstance(source, dict) and isinstance(source.get("source"), str):
            return source["source"]
        return None
    return None


def retrieval_terms(question: str, history: list[dict[str, object]]) -> str:
    follow_up = re.search(
        r"\b(he|him|his|she|her|they|them|their|it|this|that|these|those)\b",
        question,
        re.IGNORECASE,
    )
    prior_user_text = ""
    if follow_up:
        prior_turns = [
            turn
            for turn in history
            if turn.get("role") == "user" and isinstance(turn.get("content"), str)
        ]
        if prior_turns:
            prior_user_text = prior_turns[-1]["content"][:200]
    return f"{prior_user_text} {question}".strip()


def restrict_to_named_source(
    query_text: str, documents: list[Document]
) -> list[Document]:
    entity_terms = {
        term
        for term in re.findall(r"[a-zA-Z0-9]+", query_text.lower())
        if len(term) > 2 and term not in QUESTION_STOP_WORDS
    }
    if not entity_terms:
        return documents

    scores: dict[str, int] = {}
    for document in documents:
        searchable = (
            f"{document.metadata.get('source', '')} {document.page_content}"
        ).lower()
        score = sum(term in searchable for term in entity_terms)
        source = document.metadata["source"]
        scores[source] = max(scores.get(source, 0), score)
    best_score = max(scores.values(), default=0)
    if best_score == 0:
        return documents
    matching_sources = {
        source for source, score in scores.items() if score == best_score
    }
    return [
        document
        for document in documents
        if document.metadata["source"] in matching_sources
    ]


def named_sources_in_query(
    query_text: str, documents: list[Document]
) -> list[str]:
    query_terms = set(re.findall(r"[a-zA-Z0-9]+", query_text.lower()))
    source_paths: list[str] = []
    for document in documents:
        source = document.metadata["source"]
        basename = os.path.basename(source).rsplit(".", 1)[0]
        source_terms = {
            term
            for term in re.findall(r"[a-zA-Z0-9]+", basename.lower())
            if len(term) > 3 and term not in QUESTION_STOP_WORDS
        }
        if source_terms & query_terms and source not in source_paths:
            source_paths.append(source)
    return source_paths


def named_source_patterns(query_text: str) -> list[str]:
    terms = {
        term
        for term in re.findall(r"[a-zA-Z0-9]+", query_text.lower())
        if len(term) > 3 and term not in QUESTION_STOP_WORDS
    }
    return [f"%{term}%" for term in sorted(terms)]


def retrieve(
    question: str, limit: int, history: list[dict[str, object]]
) -> list[Document]:
    database_url = required_setting("DATABASE_URL")
    query_text = retrieval_terms(question, history)
    query_vector = vector_literal(
        ollama_embeddings().embed_query(f"search_query: {query_text}")
    )
    terms = list(dict.fromkeys(re.findall(r"[a-zA-Z0-9]+", query_text.lower())))
    if re.search(r"\b(full[\s-]?stack|web development|suitable)\b", query_text, re.I):
        terms.extend(
            term
            for term in (
                "frontend",
                "backend",
                "react",
                "django",
                "express",
                "database",
                "api",
                "projects",
            )
            if term not in terms
        )
    lexical_query = " | ".join(terms[:20]) or "zzzzunlikely"
    source_path = referenced_source(question, history)
    source_filter = " AND d.source_path = %s" if source_path else ""
    candidate_limit = min(max(limit * 4, 20), 80)
    with psycopg.connect(database_url) as connection:
        ensure_schema(connection)
        name_patterns = named_source_patterns(question) if not source_path else []
        if name_patterns:
            named_paths = connection.execute(
                """
                SELECT source_path
                FROM rag_documents
                WHERE lower(source_path) LIKE ANY(%s)
                ORDER BY source_path
                """,
                (name_patterns,),
            ).fetchall()
            source_paths = list(dict.fromkeys(row[0] for row in named_paths))
            if source_paths:
                named_rows = connection.execute(
                    """
                    SELECT d.source_path, c.chunk_index, c.content,
                           1 - (c.embedding <=> %s::vector) AS similarity
                    FROM rag_chunks AS c
                    JOIN rag_documents AS d ON d.id = c.document_id
                    WHERE d.source_path = ANY(%s)
                    ORDER BY d.source_path, c.chunk_index
                    LIMIT %s
                    """,
                    (query_vector, source_paths, candidate_limit),
                ).fetchall()
                return [
                    Document(
                        page_content=row[2],
                        metadata={
                            "source": row[0],
                            "chunk": row[1],
                            "similarity": float(row[3]),
                        },
                    )
                    for row in named_rows
                ]
        rows = connection.execute(
            f"""
            WITH vector_candidates AS (
                SELECT c.id,
                       row_number() OVER (
                           ORDER BY c.embedding <=> %s::vector
                       ) AS rank,
                       1 - (c.embedding <=> %s::vector) AS similarity
                FROM rag_chunks AS c
                JOIN rag_documents AS d ON d.id = c.document_id
                WHERE TRUE{source_filter}
                ORDER BY c.embedding <=> %s::vector
                LIMIT %s
            ),
            lexical_candidates AS (
                SELECT c.id,
                       row_number() OVER (
                           ORDER BY ts_rank_cd(
                               to_tsvector('simple', c.content),
                               to_tsquery('simple', %s)
                           ) DESC
                       ) AS rank,
                       0.0::double precision AS similarity
                FROM rag_chunks AS c
                JOIN rag_documents AS d ON d.id = c.document_id
                WHERE to_tsvector('simple', c.content)
                      @@ to_tsquery('simple', %s){source_filter}
                ORDER BY ts_rank_cd(
                    to_tsvector('simple', c.content),
                    to_tsquery('simple', %s)
                ) DESC
                LIMIT %s
            ),
            combined AS (
                SELECT id, rank, similarity FROM vector_candidates
                UNION ALL
                SELECT id, rank, similarity FROM lexical_candidates
            ),
            fused AS (
                SELECT id,
                       sum(1.0 / (60 + rank)) AS score,
                       max(similarity) AS similarity
                FROM combined
                GROUP BY id
            ),
            deduplicated AS (
                SELECT DISTINCT ON (md5(c.content))
                       d.source_path, c.chunk_index, c.content, fused.similarity,
                       fused.score
                FROM fused
                JOIN rag_chunks AS c ON c.id = fused.id
                JOIN rag_documents AS d ON d.id = c.document_id
                ORDER BY md5(c.content), fused.score DESC, fused.similarity DESC
            )
            SELECT source_path, chunk_index, content, similarity
            FROM deduplicated
            ORDER BY score DESC, similarity DESC
            LIMIT %s
            """,
            (
                query_vector,
                query_vector,
                *([source_path] if source_path else []),
                query_vector,
                candidate_limit,
                lexical_query,
                lexical_query,
                *([source_path] if source_path else []),
                lexical_query,
                candidate_limit,
                candidate_limit,
            ),
        ).fetchall()
    documents = [
        Document(
            page_content=row[2],
            metadata={
                "source": row[0],
                "chunk": row[1],
                "similarity": float(row[3]),
            },
        )
        for row in rows
    ]
    comparison_sources = (
        named_sources_in_query(query_text, documents) if not source_path else []
    )
    if len(comparison_sources) > 1:
        with psycopg.connect(database_url) as connection:
            comparison_rows = connection.execute(
                """
                WITH distinct_chunks AS (
                    SELECT DISTINCT ON (md5(c.content))
                           d.source_path, c.chunk_index, c.content,
                           1 - (c.embedding <=> %s::vector) AS similarity
                    FROM rag_chunks AS c
                    JOIN rag_documents AS d ON d.id = c.document_id
                    WHERE d.source_path = ANY(%s)
                    ORDER BY md5(c.content), d.source_path, c.chunk_index
                )
                SELECT source_path, chunk_index, content, similarity
                FROM distinct_chunks
                ORDER BY source_path, chunk_index
                LIMIT %s
                """,
                (query_vector, comparison_sources, min(candidate_limit, 20)),
            ).fetchall()
        return [
            Document(
                page_content=row[2],
                metadata={
                    "source": row[0],
                    "chunk": row[1],
                    "similarity": float(row[3]),
                },
            )
            for row in comparison_rows
        ]

    named_documents = restrict_to_named_source(query_text, documents)
    if named_documents and len(named_documents) < len(documents):
        source_path = named_documents[0].metadata["source"]
        with psycopg.connect(database_url) as connection:
            additional_rows = connection.execute(
                """
                SELECT d.source_path, c.chunk_index, c.content,
                       1 - (c.embedding <=> %s::vector) AS similarity
                FROM rag_chunks AS c
                JOIN rag_documents AS d ON d.id = c.document_id
                WHERE d.source_path = %s
                ORDER BY c.embedding <=> %s::vector
                LIMIT %s
                """,
                (query_vector, source_path, query_vector, candidate_limit),
            ).fetchall()
        chunks = {
            document.metadata["chunk"]: document
            for document in named_documents
        }
        for row in additional_rows:
            chunks.setdefault(
                row[1],
                Document(
                    page_content=row[2],
                    metadata={
                        "source": row[0],
                        "chunk": row[1],
                        "similarity": float(row[3]),
                    },
                ),
            )
        return [
            chunks[index]
            for index in sorted(chunks)[:limit]
        ]
    return named_documents[:limit]


def retrieve_for_chain(inputs: dict[str, object]) -> dict[str, object]:
    question = inputs["question"]
    top_k = inputs["top_k"]
    history = inputs.get("history", [])
    if (
        not isinstance(question, str)
        or not isinstance(top_k, int)
        or not isinstance(history, list)
    ):
        raise TypeError("The RAG chain received invalid retrieval input")
    documents = retrieve(question, top_k, history)
    if not documents:
        raise RuntimeError("No indexed chunks found. Ingest documents before querying.")
    return {"question": question, "documents": documents, "history": history}


def format_context(inputs: dict[str, object]) -> str:
    documents = inputs["documents"]
    if not isinstance(documents, list) or not all(
        isinstance(document, Document) for document in documents
    ):
        raise TypeError("The RAG chain received invalid document context")

    return "\n\n---\n\n".join(
        f"Document label for attribution only: "
        f"{Path(document.metadata['source']).name}\n"
        f"Document excerpt:\n{document.page_content}"
        for document in documents
    )


def create_rag_chain():
    return RunnableLambda(retrieve_for_chain) | RunnablePassthrough.assign(
        answer=create_answer_chain()
    )


def create_answer_prompt():
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You answer questions using the supplied document excerpts. Output only "
                "the answer, never your reasoning or process. Begin with the direct "
                "answer; do not say you are reviewing or analyzing anything. Combine "
                "related facts into a concise explanation instead of describing excerpts "
                "one by one. Use only facts supported by the excerpts and do not mix up "
                "different people. Each excerpt includes an internal document label; "
                "use it to keep each person's evidence separate. Never include document "
                "labels or filenames in the answer. For full-stack suitability, connect "
                "the person's "
                "actual frontend, backend, database, API, project, and deployment "
                "experience to the role; mention missing evidence briefly. Give the "
                "detail requested, normally in 2 short paragraphs and no more than "
                "180 words. Distinguish skills listed on a resume from work explicitly "
                "described in a role or project. Do not invent grades, metrics, "
                "certifications, employers, projects, or hands-on cloud experience. "
                "Never list filenames or source numbers. Do not repeat phone numbers "
                "or email addresses unless asked. If the excerpts do not contain enough "
                "evidence, state exactly what is missing. For follow-ups asking for "
                "more about a person, use the matching person's retrieved excerpts and "
                "add relevant details without restarting the whole answer. When comparing named "
                "people for a role, state the strongest fit first, compare only "
                "evidence present for each person, distinguish work experience from "
                "academic projects and self-listed skills, and explain which resume "
                "better matches the requested role.",
            ),
            (
                "human",
                "Recent conversation:\n{history}\n\n"
                "Answer style:\n{answer_style}\n\n"
                "Retrieved document context:\n{context}\n\n"
                "Latest question: {question}",
            ),
        ]
    )
    return prompt


class ThinkingFilter:
    opening = "<think>"
    closing = "</think>"

    def __init__(self) -> None:
        self.pending = ""
        self.mode = "unknown"

    def feed(self, text: str) -> str:
        if self.mode == "answer":
            opening_index = text.find(self.opening)
            if opening_index < 0:
                return text
            visible = text[:opening_index]
            self.pending = text[opening_index + len(self.opening) :]
            self.mode = "thinking"
            return visible + self.feed("")

        self.pending += text
        if self.mode == "thinking":
            closing_index = self.pending.find(self.closing)
            if closing_index < 0:
                self.pending = self.pending[-(len(self.closing) - 1) :]
                return ""
            self.pending = self.pending[closing_index + len(self.closing) :]
            self.mode = "answer"
            visible = self.pending
            self.pending = ""
            return visible

        opening_index = self.pending.find(self.opening)
        closing_index = self.pending.find(self.closing)
        if opening_index >= 0 and (
            closing_index < 0 or opening_index < closing_index
        ):
            visible = self.pending[:opening_index]
            self.pending = self.pending[opening_index + len(self.opening) :]
            self.mode = "thinking"
            return visible + self.feed("")
        if closing_index >= 0:
            self.pending = self.pending[closing_index + len(self.closing) :]
            self.mode = "answer"
            visible = self.pending
            self.pending = ""
            return visible
        return ""

    @property
    def in_thinking(self) -> bool:
        return self.mode == "thinking"

    def finish(self, completed: bool) -> str:
        if self.mode == "answer":
            visible = self.pending
            self.pending = ""
            return visible
        if not completed or self.mode == "thinking":
            self.pending = ""
            return ""
        visible = self.pending
        self.pending = ""
        return visible


def create_chat_model() -> ChatOllama:
    return ChatOllama(
        model=chat_model(),
        base_url=ollama_base_url(),
        temperature=0.1,
        reasoning=False,
        num_predict=2048,
        keep_alive=300,
        sync_client_kwargs={"timeout": 600},
    )


def create_answer_chain():
    return (
        RunnablePassthrough.assign(
            context=RunnableLambda(format_context),
            history=RunnableLambda(
                lambda inputs: format_history(
                    inputs.get("history", []), inputs.get("question", "")
                )
            ),
            answer_style=RunnableLambda(
                lambda inputs: answer_style(inputs.get("question", ""))
            ),
        )
        | create_answer_prompt()
        | create_chat_model()
    )


def answer_text(message: AIMessage) -> str:
    if not isinstance(message.content, str):
        raise RuntimeError("Ollama returned a non-text answer")
    if message.response_metadata.get("done_reason") == "length":
        raise RuntimeError("The answer reached the generation limit before completion")
    answer_filter = ThinkingFilter()
    answer = answer_filter.feed(message.content)
    answer += answer_filter.finish(completed=True)
    if answer_filter.in_thinking:
        raise RuntimeError("Ollama returned an unfinished reasoning block")
    return answer.strip()


def has_complete_ending(answer: str) -> bool:
    ending = answer.rstrip()
    while ending and ending[-1] in "\"'`)]}":
        ending = ending[:-1].rstrip()
    return bool(ending) and ending[-1] in ".!?"


def has_complete_user_answer(answer: str) -> bool:
    return (
        len(re.findall(r"\b\w+\b", answer)) >= 20
        and has_complete_ending(answer)
    )


def stream_result(
    question: str, top_k: int, history: list[dict[str, object]]
) -> None:
    result = retrieve_for_chain(
        {"question": question, "top_k": top_k, "history": history}
    )
    documents = result["documents"]
    sources = [
        {
            "source": document.metadata["source"],
            "chunk": document.metadata["chunk"],
            "similarity": document.metadata["similarity"],
        }
        for document in documents
    ]
    print(json.dumps({"type": "sources", "sources": sources}), flush=True)

    prompt_inputs = {
        **result,
        "context": format_context(result),
        "history": format_history(result.get("history", history), question),
        "answer_style": answer_style(question),
    }
    prompt_messages = create_answer_prompt().invoke(prompt_inputs).to_messages()
    answer_chunks: list[str] = []
    visible_answer_chunks: list[str] = []
    completed = False
    for attempt in range(3):
        messages = prompt_messages
        if answer_chunks:
            messages = [
                *prompt_messages,
                AIMessage(content="".join(answer_chunks)),
                HumanMessage(
                    content=(
                        "Continue the same answer exactly where it stopped. "
                        "Do not repeat or summarize what you already wrote. "
                        "Finish any incomplete sentence or list item, then stop."
                    )
                ),
            ]
        last_chunk = None
        thinking_filter = ThinkingFilter()
        stream = create_chat_model().stream(messages)
        finished_answer = False
        for chunk in stream:
            last_chunk = chunk
            token = chunk.content
            if isinstance(token, str) and token:
                answer_chunks.append(token)
                visible_token = thinking_filter.feed(token)
                if visible_token:
                    visible_answer_chunks.append(visible_token)
                    print(
                        json.dumps({"type": "token", "content": visible_token}),
                        flush=True,
                    )
            if thinking_filter.in_thinking and has_complete_user_answer(
                "".join(visible_answer_chunks).strip()
            ):
                finished_answer = True
                close_stream = getattr(stream, "close", None)
                if callable(close_stream):
                    close_stream()
                break
        if finished_answer:
            completed = True
            break
        done_reason = (
            last_chunk.response_metadata.get("done_reason")
            if last_chunk is not None
            else None
        )
        final_text = thinking_filter.finish(completed=done_reason == "stop")
        if final_text:
            visible_answer_chunks.append(final_text)
            print(json.dumps({"type": "token", "content": final_text}), flush=True)
        visible_answer = "".join(visible_answer_chunks).strip()
        if done_reason == "stop" and not thinking_filter.in_thinking:
            completed = True
            break
        if (
            not thinking_filter.in_thinking
            and has_complete_user_answer(visible_answer)
        ):
            completed = True
            break
        if done_reason != "length":
            break

    if not "".join(visible_answer_chunks).strip():
        raise RuntimeError("Ollama returned an empty answer")
    if not completed:
        print(
            json.dumps(
                {
                    "type": "error",
                    "message": "The answer could not be completed. Please ask a shorter question.",
                }
            ),
            flush=True,
        )
        return
    print(json.dumps({"type": "done"}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Retrieve document context from pgvector and ask the local Ollama model."
    )
    parser.add_argument("question", help="Question to answer from indexed documents")
    parser.add_argument(
        "--top-k",
        type=int,
        default=int(os.getenv("RAG_TOP_K", "5")),
        help="Number of similar chunks to retrieve (1-20; default: RAG_TOP_K or 5)",
    )
    parser.add_argument(
        "--json", action="store_true", help="Print the answer and sources as JSON"
    )
    parser.add_argument(
        "--stream", action="store_true", help="Print answer events as newline-delimited JSON"
    )
    parser.add_argument(
        "--history-json",
        default="[]",
        help="Recent conversation as a JSON array of user/assistant turns",
    )
    args = parser.parse_args()
    question = args.question.strip()
    if not question:
        parser.error("question must not be empty")
    if not 1 <= args.top_k <= 20:
        parser.error("--top-k must be between 1 and 20")
    try:
        history = json.loads(args.history_json)
    except json.JSONDecodeError as error:
        parser.error(f"--history-json must contain valid JSON: {error}")
    if (
        not isinstance(history, list)
        or len(history) > 8
        or not all(
            isinstance(turn, dict)
            and turn.get("role") in {"user", "assistant"}
            and isinstance(turn.get("content"), str)
            and len(turn["content"]) <= 2000
            for turn in history
        )
    ):
        parser.error(
            "--history-json must contain at most 8 user/assistant turns with text up to 2000 characters"
        )

    if args.stream:
        stream_result(question, args.top_k, history)
        return

    result = create_rag_chain().invoke(
        {"question": question, "top_k": args.top_k, "history": history}
    )
    answer = answer_text(result["answer"])
    if not answer:
        raise RuntimeError("Ollama returned an empty answer")
    documents = result["documents"]
    sources = [
        {
            "source": document.metadata["source"],
            "chunk": document.metadata["chunk"],
            "similarity": document.metadata["similarity"],
        }
        for document in documents
    ]
    if args.json:
        print(json.dumps({"answer": answer, "sources": sources}))
    else:
        print(f"\nAnswer:\n{answer}\n\nRetrieved sources:")
        for index, source in enumerate(sources, start=1):
            print(
                f"[{index}] {source['source']} (chunk {source['chunk']}; "
                f"cosine similarity {source['similarity']:.4f})"
            )


if __name__ == "__main__":
    main()
