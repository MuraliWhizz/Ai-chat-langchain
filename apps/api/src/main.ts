import { NestFactory } from '@nestjs/core';
import { AppModule } from './app.module.js';

async function bootstrap() {
  const app = await NestFactory.create(AppModule);
  const webOrigins = (
    process.env.WEB_ORIGIN ??
    'http://localhost:3000,http://127.0.0.1:3000'
  )
    .split(',')
    .map((origin) => origin.trim())
    .filter(Boolean);
  app.enableCors({
    origin: webOrigins,
  });
  await app.listen(process.env.PORT ?? 3001);
}
await bootstrap();