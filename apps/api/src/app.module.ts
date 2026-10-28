import { Module } from '@nestjs/common';
import { AppController } from './app.controller.js';
import { AppService } from './app.service.js';
import { RagController } from './rag/rag.controller.js';
import { RagService } from './rag/rag.service.js';

@Module({
  controllers: [AppController, RagController],
  providers: [AppService, RagService],
})
export class AppModule {}
