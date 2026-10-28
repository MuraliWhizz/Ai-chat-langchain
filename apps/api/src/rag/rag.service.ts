import {
  BadRequestException,
  BadGatewayException,
  Injectable,
  Logger,
  ServiceUnavailableException,
} from '@nestjs/common';
import { execFile, spawn } from 'node:child_process';
import { access, mkdir, rm, writeFile } from 'node:fs/promises';
import { randomUUID } from 'node:crypto';
import { extname, resolve } from 'node:path';
import { promisify } from 'node:util'; 
import type { Response } from 'express';

const execFileAsync = promisify(execFile);
const MAX_OUTPUT_BYTES = 1024 * 1024;
const MAX_TIMEOUT_MS = 10 * 60 * 1000;
const SUPPORTED_EXTENSIONS = new Set([
  '.docx',
  '.html',
  '.md',
  '.pdf',
  '.pptx',
  '.txt',
  '.xlsx',
]);

export interface RagSource {
  source: string;
  chunk: number;
  similarity: number;
}

export interface RagAnswer {
  answer: string;
  sources: RagSource[];
}

export interface RagHistoryTurn {
  role: 'user' | 'assistant';
  content: string;
  sources?: RagSource[];
}

export interface IngestResult {
  source: string;
  status: 'indexed' | 'unchanged';
  chunks: number;
}

@Injectable()
export class RagService {
  private readonly logger = new Logger(RagService.name);
  private readonly projectRoot = resolve(
    process.env.RAG_PROJECT_ROOT ?? resolve(process.cwd(), '../..'),
  );
  private readonly pythonPath =
    process.env.RAG_PYTHON ?? resolve(this.projectRoot, '.venv-docling/bin/python');
  private readonly servicePath = resolve(this.projectRoot, 'services/docling');

  async checkHealth(): Promise<{ status: 'ok'; service: string }> {
    await access(this.pythonPath);
    return { status: 'ok', service: 'rag' };
  }

  async ask(
    question: string,
    topK = 5,
    history: RagHistoryTurn[] = [],
  ): Promise<RagAnswer> {
    const { stdout } = await this.runPython(
      'query.py',
      [
        question,
        '--top-k',
        String(topK),
        '--history-json',
        JSON.stringify(history),
        '--json',
      ],
      MAX_TIMEOUT_MS,
    );

    try {
      const result: unknown = JSON.parse(stdout);
      if (
        typeof result !== 'object' ||
        result === null ||
        !('answer' in result) ||
        typeof result.answer !== 'string' ||
        !('sources' in result) ||
        !Array.isArray(result.sources)
      ) {
        throw new Error('Unexpected answer response shape');
      }
      const sources = result.sources;
      if (
        !sources.every(
          (source): source is RagSource =>
            typeof source === 'object' &&
            source !== null &&
            'source' in source &&
            typeof source.source === 'string' &&
            'chunk' in source &&
            typeof source.chunk === 'number' &&
            'similarity' in source &&
            typeof source.similarity === 'number',
        )
      ) {
        throw new Error('Unexpected source response shape');
      }
      return { answer: result.answer, sources };
    } catch (error) {
      this.logger.error('RAG query returned invalid JSON', error);
      throw new BadGatewayException('The RAG service returned an invalid response');
    }
  }

  stream(
    question: string,
    topK: number,
    history: RagHistoryTurn[],
    response: Response,
  ): void {
    const child = spawn(
      this.pythonPath,
      [
        resolve(this.servicePath, 'query.py'),
        question,
        '--top-k',
        String(topK),
        '--history-json',
        JSON.stringify(history),
        '--stream',
      ],
      {
        cwd: this.projectRoot,
        stdio: ['ignore', 'pipe', 'pipe'],
        windowsHide: true,
      },
    );
    let stderr = '';
    let timedOut = false;

    response.status(200).type('application/x-ndjson').set({
      'Cache-Control': 'no-cache, no-transform',
      'X-Accel-Buffering': 'no',
    });
    response.flushHeaders();

    const timeout = setTimeout(() => {
      timedOut = true;
      child.kill('SIGTERM');
    }, MAX_TIMEOUT_MS);
    timeout.unref();

    const closeChild = () => {
      if (!response.writableEnded) child.kill('SIGTERM');
    };
    response.once('close', closeChild);
    child.stdout.pipe(response, { end: false });
    child.stderr.setEncoding('utf8');
    child.stderr.on('data', (chunk: string) => {
      stderr = (stderr + chunk).slice(-2000);
    });
    child.once('error', (error) => {
      clearTimeout(timeout);
      this.logger.error('Could not start Python RAG stream', error);
      if (!response.destroyed) {
        response.write(
          `${JSON.stringify({ type: 'error', message: 'The RAG service could not start.' })}\n`,
        );
        response.end();
      }
    });
    child.once('close', (code, signal) => {
      clearTimeout(timeout);
      response.off('close', closeChild);
      if (response.destroyed || response.writableEnded) return;
      if (code !== 0) {
        const details = timedOut
          ? 'The RAG request timed out.'
          : 'The RAG service could not complete the request.';
        this.logger.error(
          `Python RAG stream failed (${signal ?? `exit code ${code ?? 'unknown'}`}): ${stderr.trim()}`,
        );
        response.write(`${JSON.stringify({ type: 'error', message: details })}\n`);
      }
      response.end();
    });
  }

  async ingest(
    file: Express.Multer.File,
  ): Promise<{ document: IngestResult }> {
    const extension = extname(file.originalname).toLowerCase();
    if (!SUPPORTED_EXTENSIONS.has(extension)) {
      throw new BadRequestException('Unsupported document extension');
    }

    const uploadsPath = resolve(this.projectRoot, 'documents/uploads');
    await mkdir(uploadsPath, { recursive: true });
    const storedPath = resolve(uploadsPath, `${randomUUID()}${extension}`);

    try {
      await writeFile(storedPath, file.buffer, { flag: 'wx' });
      const { stdout } = await this.runPython(
        'ingest.py',
        ['--json', storedPath],
        MAX_TIMEOUT_MS,
      );
      const results: unknown = JSON.parse(stdout);
      if (
        !Array.isArray(results) ||
        results.length !== 1 ||
        typeof results[0] !== 'object' ||
        results[0] === null ||
        !('source' in results[0]) ||
        typeof results[0].source !== 'string' ||
        !('status' in results[0]) ||
        (results[0].status !== 'indexed' && results[0].status !== 'unchanged') ||
        !('chunks' in results[0]) ||
        typeof results[0].chunks !== 'number'
      ) {
        throw new Error('Unexpected ingestion response shape');
      }
      return {
        document: {
          source: results[0].source,
          status: results[0].status,
          chunks: results[0].chunks,
        },
      };
    } catch (error) {
      await rm(storedPath, { force: true });
      if (error instanceof BadGatewayException) {
        throw error;
      }
      this.logger.error('Document ingestion failed', error);
      throw new BadGatewayException(
        'Document ingestion failed. Check that PostgreSQL, pgvector, and Ollama are available.',
      );
    }
  }

  private async runPython(
    script: string,
    args: string[],
    timeout: number,
  ): Promise<{ stdout: string; stderr: string }> {
    try {
      return await execFileAsync(
        this.pythonPath,
        [resolve(this.servicePath, script), ...args],
        {
          cwd: this.projectRoot,
          encoding: 'utf8',
          maxBuffer: MAX_OUTPUT_BYTES,
          timeout,
          windowsHide: true,
        },
      );
    } catch (error) {
      const details =
        typeof error === 'object' && error !== null && 'stderr' in error
          ? String(error.stderr).trim().slice(-2000)
          : error instanceof Error
            ? error.message
            : String(error);
      this.logger.error(`Python RAG process failed: ${details}`);
      throw new ServiceUnavailableException(
        'The RAG service could not complete the request. Check its Python environment, PostgreSQL, and Ollama.',
      );
    }
  }
}
