import { readFileSync } from 'fs';

import { Controller, Param, Post } from '@nestjs/common';
import OpenAI from 'openai';

const openAIClient = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});
const prompt = readFileSync('./prompt', 'utf-8');

interface Node {
  id: string;
  name: string;
  arguments: Record<string, string>;
  connect_to: string;
}

@Controller('code')
export class CodeController {
  @Post()
  async generate(@Param('nodes') nodes: Node[]): Promise<string> {
    const response = await openAIClient.chat.completions.create({
      model: 'gpt-4.1-mini',
      messages: [
        {
          role: 'system',
          content: prompt,
        },
        {
          role: 'user',
          content: JSON.stringify(nodes, null, 2),
        },
      ],
    });
    const code = response.choices[0].message.content;
    return code ?? 'No code generated';
  }
}
