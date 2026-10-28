import { BadRequestException } from '@nestjs/common';
import { Test, TestingModule } from '@nestjs/testing';
import { RagController } from './rag.controller.js';
import { RagService } from './rag.service.js';

describe('RagController', () => {
  let controller: RagController;
  let ragService: {
    ask: ReturnType<typeof vi.fn>;
    checkHealth: ReturnType<typeof vi.fn>;
    ingest: ReturnType<typeof vi.fn>;
    stream: ReturnType<typeof vi.fn>;
  };

  beforeEach(async () => {
    ragService = {
      ask: vi.fn(),
      checkHealth: vi.fn(),
      ingest: vi.fn(),
      stream: vi.fn(),
    };
    const module: TestingModule = await Test.createTestingModule({
      controllers: [RagController],
      providers: [{ provide: RagService, useValue: ragService }],
    }).compile();
    controller = module.get(RagController);
  });

  it('forwards a valid question and the default topK', async () => {
    ragService.ask.mockResolvedValue({ answer: 'Answer', sources: [] });

    await expect(controller.chat({ question: '  What is this?  ' })).resolves.toEqual({
      answer: 'Answer',
      sources: [],
    });
    expect(ragService.ask).toHaveBeenCalledWith('What is this?', 5, []);
  });

  it('forwards a requested topK', async () => {
    ragService.ask.mockResolvedValue({ answer: 'Answer', sources: [] });

    await controller.chat({ question: 'Question', topK: 3 });

    expect(ragService.ask).toHaveBeenCalledWith('Question', 3, []);
  });

  it('forwards a valid streaming chat request', () => {
    const response = {} as Parameters<typeof controller.streamChat>[1];
    const history = [
      { role: 'assistant', content: 'Muralidhar is a student.' },
    ];

    controller.streamChat(
      { question: ' Question ', topK: 2, history },
      response,
    );

    expect(ragService.stream).toHaveBeenCalledWith('Question', 2, history, response);
  });

  it.each([
    undefined,
    {},
    { question: '' },
    { question: '  ' },
    { question: 42 },
    { question: 'q'.repeat(4001) },
    { question: 'Question', topK: 0 },
    { question: 'Question', topK: 21 },
    { question: 'Question', topK: 1.5 },
  ])('rejects invalid chat request %#', async (body) => {
    await expect(controller.chat(body)).rejects.toBeInstanceOf(
      BadRequestException,
    );
    expect(ragService.ask).not.toHaveBeenCalled();
  });

  it.each([
    undefined,
    {},
    { question: '' },
    { question: 'Question', topK: 0 },
  ])('rejects invalid streaming chat request %#', (body) => {
    expect(() =>
      controller.streamChat(
        body,
        {} as Parameters<typeof controller.streamChat>[1],
      ),
    ).toThrow(BadRequestException);
    expect(ragService.stream).not.toHaveBeenCalled();
  });

  it.each([
    { question: 'Question', history: 'invalid' },
    {
      question: 'Question',
      history: Array(9).fill({ role: 'user', content: 'hi' }),
    },
    { question: 'Question', history: [{ role: 'system', content: 'bad' }] },
    { question: 'Question', history: [{ role: 'user', content: 'x'.repeat(2001) }] },
    {
      question: 'Question',
      history: [
        {
          role: 'assistant',
          content: 'Answer',
          sources: [{ source: 'guide.pdf', chunk: -1, similarity: 0.9 }],
        },
      ],
    },
  ])('rejects invalid conversation history %#', async (body) => {
    await expect(controller.chat(body)).rejects.toBeInstanceOf(
      BadRequestException,
    );
    expect(ragService.ask).not.toHaveBeenCalled();
  });

  it('rejects uploads with no file', async () => {
    await expect(controller.upload(undefined)).rejects.toBeInstanceOf(
      BadRequestException,
    );
  });
});
