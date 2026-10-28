# NestJS API

The API calls the existing Python RAG service in `services/docling`. Start it
from `apps/api` with the repository's `.venv-docling` environment already set
up, and keep PostgreSQL/pgvector and Ollama running.

```bash
npm install
npm run start:dev
```

By default, the API expects the project root at `../..` from `apps/api` and the
Python executable at `.venv-docling/bin/python` under that root. Override these
locations with `RAG_PROJECT_ROOT` and `RAG_PYTHON` if needed.

## Endpoints

- `GET /health` checks that the configured Python executable is available.
- `POST /chat` accepts JSON such as
  `{"question":"What does the architecture document describe?","topK":5}`.
  `topK` is optional and must be between 1 and 20. The response contains an
  answer and the retrieved source paths/chunks/similarity scores.
- `POST /chat/stream` accepts the same JSON and streams newline-delimited JSON
  events. A `sources` event is sent before `token` events; a final `done` event
  marks successful completion. An `error` event indicates a failed request.
  Both chat endpoints accept an optional `history` array with up to 8 recent
  user/assistant turns (2000 characters each) to support follow-up questions.
  Up to 5 prior source references per assistant turn can be included to resolve
  follow-ups such as "according to source 2".
- `POST /documents` accepts `multipart/form-data` with a `file` field. Supported
  extensions are PDF, DOCX, PPTX, XLSX, HTML, Markdown, and plain text. Files
  are limited to 25 MB and are parsed and indexed before a success response.

The API listens on port 3001 by default and allows the Next.js origins
`http://localhost:3000` and `http://127.0.0.1:3000`. Set `WEB_ORIGIN` (a
comma-separated list) or `PORT` to override these defaults.
