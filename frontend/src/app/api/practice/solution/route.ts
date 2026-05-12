import Anthropic from '@anthropic-ai/sdk';
import { NextRequest } from 'next/server';
import {
  SOLUTION_SYSTEM_PROMPT,
  buildSolutionUserMessage,
} from '@/lib/practice-prompts';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

interface Body {
  brief?: string;
  userAttempt?: string;
}

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

  const brief = (body.brief ?? '').trim();
  const userAttempt = (body.userAttempt ?? '').trim();

  if (!brief) {
    return Response.json(
      { error: 'Missing `brief`. Generate a brief first.' },
      { status: 400 },
    );
  }

  const client = new Anthropic({ apiKey });
  const userMessage = buildSolutionUserMessage(brief, userAttempt);

  const stream = client.messages.stream({
    model: 'claude-opus-4-7',
    max_tokens: 16000,
    thinking: { type: 'adaptive' },
    output_config: { effort: 'high' },
    system: [
      {
        type: 'text',
        text: SOLUTION_SYSTEM_PROMPT,
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
