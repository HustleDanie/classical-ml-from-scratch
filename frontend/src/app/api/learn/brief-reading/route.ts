import Anthropic from '@anthropic-ai/sdk';
import { NextRequest } from 'next/server';
import {
  BRIEF_READING_SYSTEM_PROMPT,
  BRIEF_READING_RESPONSE_SCHEMA,
  buildBriefReadingUserMessage,
  type Complexity,
  type ScenarioType,
} from '@/lib/practice-prompts';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';
export const maxDuration = 300;

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

  try {
    const response = await client.messages.create({
      model: 'claude-opus-4-7',
      max_tokens: 16000,
      thinking: { type: 'adaptive' },
      output_config: {
        effort: 'high',
        format: {
          type: 'json_schema',
          schema: BRIEF_READING_RESPONSE_SCHEMA,
        },
      },
      system: [
        {
          type: 'text',
          text: BRIEF_READING_SYSTEM_PROMPT,
          cache_control: { type: 'ephemeral' },
        },
      ],
      messages: [{ role: 'user', content: userMessage }],
    });

    const textBlock = response.content.find((b) => b.type === 'text');
    if (!textBlock || textBlock.type !== 'text') {
      return Response.json(
        { error: 'Model returned no text content.' },
        { status: 502 },
      );
    }

    const parsed = JSON.parse(textBlock.text);
    return Response.json(parsed);
  } catch (err) {
    const message = err instanceof Error ? err.message : 'Unknown error';
    return Response.json({ error: message }, { status: 500 });
  }
}
