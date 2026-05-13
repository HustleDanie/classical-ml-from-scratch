import Anthropic from '@anthropic-ai/sdk';
import { NextRequest } from 'next/server';
import {
  BRIEF_SYSTEM_PROMPT,
  BRIEF_RESPONSE_SCHEMA,
  buildBriefUserMessage,
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

const SUBMIT_TOOL_NAME = 'submit_brief_and_dataset';

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
  const userMessage = buildBriefUserMessage(type, complexity);

  try {
    // Forced tool use is the reliable structured-output pattern: the model
    // MUST call this tool, and its `input` argument is the validated JSON.
    const response = await client.messages.create({
      model: 'claude-opus-4-7',
      max_tokens: 12000,
      thinking: { type: 'adaptive' },
      output_config: { effort: 'medium' },
      tools: [
        {
          name: SUBMIT_TOOL_NAME,
          description:
            'Submit the generated ML business brief and its matching synthetic dataset. Use this tool exactly once with the full payload.',
          input_schema: BRIEF_RESPONSE_SCHEMA as Anthropic.Tool['input_schema'],
        },
      ],
      tool_choice: { type: 'tool', name: SUBMIT_TOOL_NAME },
      system: [
        {
          type: 'text',
          text: BRIEF_SYSTEM_PROMPT,
          cache_control: { type: 'ephemeral' },
        },
      ],
      messages: [{ role: 'user', content: userMessage }],
    });

    const toolBlock = response.content.find((b) => b.type === 'tool_use');
    if (!toolBlock || toolBlock.type !== 'tool_use') {
      console.error('[practice/brief] no tool_use block in response', response.content);
      return Response.json(
        { error: 'Model did not return a structured response.' },
        { status: 502 },
      );
    }
    // toolBlock.input is already a parsed object matching BRIEF_RESPONSE_SCHEMA.
    return Response.json(toolBlock.input);
  } catch (err) {
    const message = err instanceof Error ? err.message : 'Unknown error';
    console.error('[practice/brief] error:', message);
    return Response.json({ error: message }, { status: 500 });
  }
}
