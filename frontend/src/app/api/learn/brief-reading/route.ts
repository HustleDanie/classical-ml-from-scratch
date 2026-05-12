import Anthropic from '@anthropic-ai/sdk';
import { NextRequest } from 'next/server';
import {
  BRIEF_READING_SYSTEM_PROMPT,
  buildBriefReadingUserMessage,
  type Complexity,
  type ScenarioType,
} from '@/lib/practice-prompts';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

interface Body {
  type?: ScenarioType;
  complexity?: Complexity;
}

const VALID_TYPES: ScenarioType[] = ['classification', 'regression', 'random'];
const VALID_COMPLEXITY: Complexity[] = ['easy', 'medium', 'hard', 'random'];

export async function POST(req: NextRequest) {
  const apiKey = process.env.ANTHROPIC_API_KEY;
  if (!apiKey) {
    return Response.json(
      { error: 'ANTHROPIC_API_KEY is not configured on the server.' },
      { status: 503 },
    );
  }

  let body: Body;
  try {
    body = await req.json();
  } catch {
    return Response.json({ error: 'Invalid JSON body.' }, { status: 400 });
  }

  const type: ScenarioType = VALID_TYPES.includes(body.type as ScenarioType)
    ? (body.type as ScenarioType)
    : 'random';
  const complexity: Complexity = VALID_COMPLEXITY.includes(body.complexity as Complexity)
    ? (body.complexity as Complexity)
    : 'medium';

  const client = new Anthropic({ apiKey });
  const userMessage = buildBriefReadingUserMessage(type, complexity);

  const stream = client.messages.stream({
    model: 'claude-opus-4-7',
    max_tokens: 6000,
    thinking: { type: 'adaptive' },
    output_config: { effort: 'high' },
    system: [
      {
        type: 'text',
        text: BRIEF_READING_SYSTEM_PROMPT,
        cache_control: { type: 'ephemeral' },
      },
    ],
    messages: [{ role: 'user', content: userMessage }],
  });

  const encoder = new TextEncoder();
  const responseStream = new ReadableStream<Uint8Array>({
    async start(controller) {
      try {
        for await (const event of stream) {
          if (
            event.type === 'content_block_delta' &&
            event.delta.type === 'text_delta'
          ) {
            controller.enqueue(encoder.encode(event.delta.text));
          }
        }
        controller.close();
      } catch (err) {
        const message = err instanceof Error ? err.message : 'Unknown error';
        controller.enqueue(encoder.encode(`\n\n[ERROR] ${message}`));
        controller.close();
      }
    },
  });

  return new Response(responseStream, {
    headers: {
      'Content-Type': 'text/plain; charset=utf-8',
      'Cache-Control': 'no-store',
      'X-Accel-Buffering': 'no',
    },
  });
}
