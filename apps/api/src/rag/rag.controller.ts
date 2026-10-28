import {
  BadRequestException,
  Body,
  Controller,
  Get,
  Post,
  Res,
  UploadedFile,
  UseInterceptors,
} from '@nestjs/common';
import { FileInterceptor } from '@nestjs/platform-express';
import { extname } from 'node:path';
import { memoryStorage } from 'multer';
import type { Response } from 'express';
import { RagService, type RagSource } from './rag.service.js';

const MAX_UPLOAD_BYTES = 25 * 1024 * 1024;
const SUPPORTED_EXTENSIONS = new Set([
  '.docx',
  '.html',
  '.md',
  '.pdf',
  '.pptx',
  '.txt',
  '.xlsx',
]);

interface ChatHistoryTurn {
  role: 'user' | 'assistant';
  content: string;
  sources?: RagSource[];
}

function isRagSource(value: unknown): value is RagSource {
  return (
    typeof value === 'object' &&
    value !== null &&
    'source' in value &&
    typeof value.source === 'string' &&
    value.source.length <= 2048 &&
    'chunk' in value &&
    typeof value.chunk === 'number' &&
    Number.isInteger(value.chunk) &&
    value.chunk >= 0 &&
    'similarity' in value &&
    typeof value.similarity === 'number' &&
    Number.isFinite(value.similarity)
  );
}

function parseHistory(value: unknown): ChatHistoryTurn[] {
  if (value === undefined) return [];
  if (!Array.isArray(value) || value.length > 8) {
    throw new BadRequestException('history must contain at most 8 conversation turns');
  }

  let totalCharacters = 0;
  return value.map((turn) => {
    if (
      typeof turn !== 'object' ||
      turn === null ||
      !('role' in turn) ||
      (turn.role !== 'user' && turn.role !== 'assistant') ||
      !('content' in turn) ||
      typeof turn.content !== 'string' ||
      turn.content.trim().length === 0 ||
      turn.content.length > 2000
    ) {
      throw new BadRequestException('history contains an invalid conversation turn');
    }

    totalCharacters += turn.content.length;
    if (totalCharacters > 12000) {
      throw new BadRequestException('history is too long');
    }

    let sources: RagSource[] | undefined;
    if ('sources' in turn && turn.sources !== undefined) {
      if (
        turn.role !== 'assistant' ||
        !Array.isArray(turn.sources) ||
        turn.sources.length > 5 ||
        !turn.sources.every(isRagSource)
      ) {
        throw new BadRequestException('history contains invalid source references');
      }
      sources = turn.sources;
    }
    return {
      role: turn.role,
      content: turn.content,
      ...(sources ? { sources } : {}),
    };
  });
}

function parseChatRequest(body: unknown): {
  question: string;
  topK: number;
  history: ChatHistoryTurn[];
} {
  if (typeof body !== 'object' || body === null || !('question' in body)) {
    throw new BadRequestException('question is required');
  }
  const { question } = body;
  if (
    typeof question !== 'string' ||
    question.trim().length === 0 ||
    question.length > 4000
  ) {
    throw new BadRequestException(
      'question must be a non-empty string of at most 4000 characters',
    );
  }

  let topK = 5;
  if ('topK' in body && body.topK !== undefined) {
    if (
      typeof body.topK !== 'number' ||
      !Number.isInteger(body.topK) ||
      body.topK < 1 ||
      body.topK > 20
    ) {
      throw new BadRequestException('topK must be an integer between 1 and 20');
    }
    topK = body.topK;
  }
  const history =
    'history' in body ? parseHistory(body.history) : [];
  return { question: question.trim(), topK, history };
}

@Controller()
export class RagController {
  constructor(private readonly ragService: RagService) {}

  @Get('health')
  async health() {
    return this.ragService.checkHealth();
  }

  @Post('chat')
  async chat(@Body() body: unknown) {
    const request = parseChatRequest(body);
    return this.ragService.ask(request.question, request.topK, request.history);
  }

  @Post('chat/stream')
  streamChat(@Body() body: unknown, @Res() response: Response): void {
    const request = parseChatRequest(body);
    this.ragService.stream(
      request.question,
      request.topK,
      request.history,
      response,
    );
  }

  @Post('documents')
  @UseInterceptors(
    FileInterceptor('file', {
      storage: memoryStorage(),
      limits: { fileSize: MAX_UPLOAD_BYTES, files: 1 },
      fileFilter: (_request, file, callback) => {
        if (!SUPPORTED_EXTENSIONS.has(extname(file.originalname).toLowerCase())) {
          callback(new BadRequestException('Unsupported document extension'), false);
          return;
        }
        callback(null, true);
      },
    }),
  )
  async upload(@UploadedFile() file: Express.Multer.File | undefined) {
    if (!file) {
      throw new BadRequestException(
        'A supported document is required in the "file" field',
      );
    }
    return this.ragService.ingest(file);
  }
}
