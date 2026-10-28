# AI Knowledge Assistant

This project runs a local LangChain RAG pipeline: Docling parses documents,
Qwen3 embeddings are stored in PostgreSQL/pgvector, and LangChain retrieves
relevant chunks and sends them to Qwen3 4B. NestJS exposes the chat and document
ingestion endpoints to the Next.js UI.

## Prerequisites

- Python 3.10 or newer and the project's `.venv-docling` environment
- PostgreSQL with the `vector` (pgvector) extension available to the database
- Ollama running locally with `qwen3-embedding:0.6b` and `qwen3:4b` available

## Install project dependencies

From the project root, install the root, API, web, and Docling dependencies in
sequence with one command:

```bash
npm run cmd -- install --all
```

Each install runs in the same terminal and displays its output. The command
stops and returns a failure if any step fails. It creates `.venv-docling` if
needed and installs `services/docling/requirements.txt` into that environment.

Create or update the Python environment and configure the database:

```bash
source .venv-docling/bin/activate
python -m pip install -r services/docling/requirements.txt
cp .env.example .env
```

Edit `.env` and set `DATABASE_URL` to the PostgreSQL database where you want
the RAG tables created. To run PostgreSQL with pgvector and Ollama in Docker,
start the dependencies from the project root:

```bash
docker compose up -d
docker compose exec ollama ollama pull qwen3-embedding:0.6b
docker compose exec ollama ollama pull qwen3:4b
```

The Compose defaults create the `ai_assistant` database and user with password
`ai_assistant`; set `DATABASE_URL=postgresql://ai_assistant:ai_assistant@127.0.0.1:5432/ai_assistant`
in `.env` to use them. You can override the database name, user, password, and
host port with `POSTGRES_DB`, `POSTGRES_USER`, `POSTGRES_PASSWORD`, and
`POSTGRES_PORT`. Set `OLLAMA_PORT` to change Ollama's host port and update
`OLLAMA_BASE_URL` accordingly. Stop the dependencies with `docker compose down`;
their data is retained in named Docker volumes.

Do not commit `.env`. The scripts also read
`apps/api/.env` as a fallback when a root `.env` is absent or does not define a
setting.

For example, the configured database must have permission to install pgvector
or have the extension installed by an administrator:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

The first ingest or query command creates `rag_documents` and `rag_chunks`,
including a 1024-dimensional vector column and cosine-similarity index.

## Index documents

Put source files under `documents/`, or pass a file or directory path directly.
Supported formats are PDF, DOCX, PPTX, XLSX, HTML, Markdown, and plain text.

```bash
python services/docling/ingest.py documents/
python services/docling/ingest.py /path/to/architecture.docx
```

Unchanged files are skipped using a SHA-256 content hash. Re-ingesting a changed
file replaces its chunks in a single database transaction.

## Ask a question

```bash
python services/docling/query.py "What does the architecture document describe?"
```

LangChain's Ollama integration creates embeddings with the existing
`search_document:` and `search_query:` prefixes. The RAG chain retrieves chunks
from the existing `rag_chunks` pgvector table, then uses LangChain's Ollama chat
integration to answer with that context. This keeps previously indexed vectors
and does not require a database migration. The command prints the answer and
retrieved source list. Set `RAG_TOP_K` to change the default number of retrieved
chunks (valid range: 1–20).

The chat UI uses `POST /chat/stream` to show Qwen's answer as tokens arrive,
instead of waiting for the entire response. `POST /chat` remains available for
clients that need a single JSON response. Recent conversation turns are passed
to retrieval and generation to support follow-up questions. Retrieved source
metadata stays available internally for that context but is not shown as source
cards in the chat.

### Retrieval and answer-quality improvements

- **Hybrid retrieval:** combines pgvector similarity search with PostgreSQL
  full-text search, fuses the rankings, and removes duplicate excerpts.
- **Person-aware retrieval:** when a question names someone in an indexed
  document's filename, retrieve that document's chunks directly instead of
  relying on generic similarity rankings. This helps keep similarly named
  people and unrelated documents out of the answer.
- **Complete resume comparisons:** when a question names multiple indexed
  people, retrieve their matching documents together so the model can compare
  the evidence from both resumes.
- **Follow-up handling:** retain recent user context for pronoun-based
  questions. If a follow-up explicitly names a person from the prior answer,
  use that person's document and avoid carrying the earlier comparison prompt
  into the new answer.
- **Grounded, focused answers:** instruct Qwen to answer directly, combine
  relevant facts instead of narrating excerpts, distinguish resume-listed
  skills from described experience, avoid unsupported claims, and adapt answer
  length when the user asks for a brief response.
- **Private source presentation:** source metadata and internal document labels
  help attribute facts to the correct resume, but filenames and source cards
  are not displayed in chat.
- **Reliable streaming:** filter Qwen3 `<think>` content from the visible
  answer, continue generation after token-limit stops, and emit completion
  only after a normal stop or a sentence-complete answer. Incomplete responses
  are reported as errors rather than silently presented as finished.
- **Responsive chat UI:** stream answer text as it is generated and show one
  assistant message with an inline search indicator, avoiding duplicate
  assistant bubbles while retrieval is running.
- **Lower inference overhead:** keep Ollama models loaded for five minutes,
  batch embedding requests during ingestion, and avoid loading Docling's
  conversion runtime for chat-only queries.

Retrieval combines pgvector similarity with PostgreSQL full-text keyword
matching, fuses the ranked results, and removes duplicate indexed excerpts.
Named-person and comparison questions use filename matching to select the
relevant indexed document or documents before the model sees the context.
Explicit follow-ups such as "according to source 2" are restricted to that
previously retrieved document when its reference is available. If Ollama stops
because it reaches its token limit, the stream asks it to continue; a response
is marked complete only after a normal stop or a sentence-complete answer.

Chat inference requests Qwen3's non-reasoning mode and caps each generation
attempt at 2048 tokens. Any Qwen3 `<think>` content is filtered from the chat;
the stream is marked complete after a normal stop or a sentence-complete answer
if Qwen3 starts reasoning again after finishing its response. Ollama keeps the
embedding and chat models loaded for five minutes after use to avoid reloading
them between nearby requests. Document ingestion batches embeddings through
LangChain, and chat queries no longer import Docling's document-conversion
runtime. Retrieved excerpt prompts include an internal filename label to help
the model attribute evidence to the right document; filenames and source cards
are not displayed in chat.

Optional local service settings:

```dotenv
OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_EMBED_MODEL=qwen3-embedding:0.6b
OLLAMA_CHAT_MODEL=qwen3:4b
```

## Start the backend API and chat UI together

From the project root, start both the NestJS API and Next.js web app with:

```bash
npm start
```

Open [http://localhost:3000](http://localhost:3000) for the chat UI; the NestJS
API runs on [http://localhost:3001](http://localhost:3001). Press Ctrl+C to stop
both processes. `npm run dev` is an equivalent alias. The launcher runs each app
from its own directory and does not require a root `npm install`.

The API expects the root `.venv-docling`, PostgreSQL, and Ollama to remain
available. See [apps/api/README.md](./apps/api/README.md) for endpoint details.
You can still run either app individually with `npm run dev:api` or
`npm run dev:web`.

To use different development ports, set `API_PORT` and/or `WEB_PORT`, for
example `API_PORT=3101 WEB_PORT=3100 npm run dev`.
