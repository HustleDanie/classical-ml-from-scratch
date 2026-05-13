/**
 * System prompts for the three Claude API endpoints.
 *
 *   /api/practice/brief         — returns JSON: { brief, dataset }
 *   /api/practice/solution      — streams text: 17-step walkthrough + critique
 *   /api/learn/brief-reading    — returns JSON: { brief, dataset, phases }
 *
 * Brief + dataset are co-generated so the dataset *matches* the brief. The
 * structured-output schema (declared via `output_config.format` on the API
 * route side) guarantees the response shape; this prompt teaches Claude
 * the content discipline.
 *
 * System prompts are stable across calls so the Anthropic prompt cache
 * (5-min TTL) cuts cost ~90% on the cached prefix for repeat traffic.
 */

import { DATASET_JSON_SCHEMA } from './dataset';

export type ScenarioType = 'classification' | 'regression' | 'random';
export type Complexity = 'easy' | 'medium' | 'hard' | 'random';

const COMPLEXITY_DIRECTIVES: Record<Exclude<Complexity, 'random'>, string> = {
  easy: `EASY: a clean tabular problem with balanced classes / well-behaved target, modest data size (1K–50K rows, 10–30 features), no special constraints. A learner can apply the standard pipeline and get a good model.`,
  medium: `MEDIUM: introduce ONE meaningful complication — moderate class imbalance (85/15 to 95/5), heavy-tailed regression target, mild missingness, OR a deployment constraint like batch latency. Data size 50K–1M rows. Should require the learner to make at least one non-default decision.`,
  hard: `HARD: stack TWO OR THREE constraints from this list — extreme imbalance (<1%), real-time inference (<100ms), regulated context (ECOA / FDA / EEOC), heavy-tailed target with long-tail decisions, mixed types with informative missingness, adversarial drift, fairness audit required, calibrated probabilities feeding cost math, or temporal split with regime change.`,
};

const TYPE_DIRECTIVES: Record<Exclude<ScenarioType, 'random'>, string> = {
  classification:
    'A binary or multiclass classification problem. State the positive-class definition explicitly and the imbalance level.',
  regression:
    'A regression problem on a numeric target. State the target distribution shape (skew, range, censoring if any) explicitly.',
};

/* -----------------------------------------------------------------------
 * Shared dataset-generation discipline (referenced by both endpoints below).
 * --------------------------------------------------------------------- */

const DATASET_RULES = `Dataset rules:
- 15–30 rows, 5–12 columns. Compact enough for a static preview, real enough to run pandas against.
- Include the target column the brief asks to predict; the column's distribution should match the brief's stated rate (e.g. ~7% positive class, or skewed regression target).
- Reflect the features the brief mentions. If the brief says "complaint logs", add a free-text column. If it mentions "informative missingness on field X", leave X blank in 60-80% of rows.
- Include at least one realistic LEAKER column that a Phase-2 audit would catch (e.g., inspector_id when judgement varies; chargeback_flag for fraud). The brief should hint at it.
- column names: snake_case, lowercase. e.g. \`days_since_last_inspection\`, \`is_critical\`.
- type: one of "number" | "string" | "boolean" | "date".
- rows: every cell as a STRING (numbers stringified, booleans as "true"/"false", dates as YYYY-MM-DD). Use empty string "" for blank/missing.
- All values plausible: real-looking IDs, dates in a coherent date range, dollar amounts in a believable range for the domain.
- Don't reuse the restaurant-inspection domain — pick something fresh.`;

/* -----------------------------------------------------------------------
 * BRIEF generation — returns { brief, dataset }
 * --------------------------------------------------------------------- */

export const BRIEF_SYSTEM_PROMPT = `You generate a realistic ML business brief AND a small matching dataset for a learner.

Output is a JSON object with shape:
{
  "brief": "<markdown string, 200-400 words>",
  "dataset": { filename, description, columns: [...], rows: [...] }
}

The brief markdown template (use VERBATIM, fill the bracketed parts):

# [Scenario title — descriptive, e.g. "Hospital Readmission Risk Scoring"]

> **Complexity:** [one-line callout naming the 2-3 hardest constraints — e.g. "Moderate imbalance (~9%), regulated under HIPAA, real-time at discharge, mixed types with informative missingness."]

---

## The Brief

[2-4 paragraphs from a stakeholder's voice. Include:
- The business / domain (specific: "a regional grocery chain", not "a company")
- The exact prediction target and what business decision it drives
- Data shape: row count, feature count, target distribution
- Hard constraints: latency budget, regulatory regime, cost asymmetry
- What "good" looks like: the primary metric AND the threshold a stakeholder cares about]

[Optional final paragraph: a "this is harder than it sounds because…" line that sets up the trap learners typically fall into.]

Brief rules:
- DO NOT include any solution, algorithm recommendation, pipeline steps, or "what to do" advice.
- DO NOT include section headers beyond "# Title" and "## The Brief".
- Use realistic numbers (plausible row counts, percentages, dollar amounts).
- Voice: a real stakeholder — concrete, slightly informal, with implicit urgency.
- 200-400 words total.

${DATASET_RULES}

The dataset MUST match the brief: same domain, target column matches the prediction target, features reflect the brief's mentioned columns.`;

export function buildBriefUserMessage(type: ScenarioType, complexity: Complexity): string {
  const resolvedType =
    type === 'random' ? (Math.random() < 0.5 ? 'classification' : 'regression') : type;
  const resolvedComplexity =
    complexity === 'random'
      ? (['easy', 'medium', 'hard'] as const)[Math.floor(Math.random() * 3)]
      : complexity;

  return `Generate a fresh problem brief + matching dataset.

Type: ${TYPE_DIRECTIVES[resolvedType]}

Complexity: ${COMPLEXITY_DIRECTIVES[resolvedComplexity]}

Pick an industry / domain at random — vary it across calls (don't default to fraud, churn, housing, or restaurant inspections). Make the brief specific and concrete, and produce a 15-30 row matching dataset.`;
}

export const BRIEF_RESPONSE_SCHEMA = {
  type: 'object',
  required: ['brief', 'dataset'],
  properties: {
    brief: { type: 'string', minLength: 200 },
    dataset: DATASET_JSON_SCHEMA,
  },
  additionalProperties: false,
};

/* -----------------------------------------------------------------------
 * SOLUTION generation (text-streaming, unchanged)
 * --------------------------------------------------------------------- */

export const SOLUTION_SYSTEM_PROMPT = `You are an expert ML engineer reviewing a learner's attempt at a business ML problem.

Your output has TWO sections, in this exact order:

# Model Answer

A complete 17-step expert walkthrough using these section headings VERBATIM:

## Step 1: Define the Problem Type
## Step 2: Understand the Data
## Step 3: Exploratory Data Analysis (EDA)
## Step 4: Data Cleaning
## Step 5: Feature Engineering
## Step 6: Feature Selection
## Step 7: Preprocessing
## Step 8: Train/Test Split
## Step 9: Baseline
## Step 10: Try Multiple Models with CV
## Step 11: Handle Imbalance / Target Transformation
## Step 12: Hyperparameter Tuning
## Step 13: Threshold Tuning / Calibration
## Step 14: Ensemble / Stacking
## Step 15: Final Evaluation on Test Set
## Step 16: Explainability
## Step 17: Deployment Considerations

End with a markdown table "Summary: What Made This Expert-Level" comparing "Beginner Would Do" vs "Expert Did" (8-12 rows).

---

# Compared to Your Attempt

A 3-paragraph critique:
1. What you got right — quote the user's reasoning briefly.
2. What you missed — the 1-3 most important gaps with WHY each matters.
3. One thing to internalise — a transferable lesson.

If the user submitted nothing, say so plainly and tell them what to think about next time.

Style:
- Use markdown: headings, code blocks (runnable Python), tables, bullets.
- Don't use emojis. Don't preface with "Sure, here's…" — start with "# Model Answer".`;

export function buildSolutionUserMessage(brief: string, userAttempt: string): string {
  return `# The Brief

${brief}

---

# My Attempt

${userAttempt.trim() || '(The user submitted no attempt. Note this in the critique and tell them to try harder next time.)'}

---

Produce the model answer, then the critique. Use the exact format from the system prompt.`;
}

/* -----------------------------------------------------------------------
 * BRIEF READING — returns { brief, dataset, phases }
 * --------------------------------------------------------------------- */

export const BRIEF_READING_SYSTEM_PROMPT = `You generate a complete teaching artefact: a realistic ML business brief, a small matching dataset, AND a per-phase dissection that pulls verbatim phrases from the brief to drive decisions in each of the 7 pipeline phases.

Output is a JSON object:
{
  "brief": "<markdown string>",
  "dataset": { filename, description, columns: [...], rows: [...] },
  "phases": [
    {
      "num": "01",
      "title": "Understand the Problem",
      "intro": "<one-line: what you're extracting in this phase>",
      "signals": [
        { "quote": "<verbatim phrase from the brief>", "implication": "<the concrete decision that phrase forces>" },
        ...
      ]
    },
    ... (exactly 7 phase blocks, num "01" through "07")
  ]
}

The brief uses the same markdown template as a normal brief generation (# Title, > Complexity callout, ## The Brief).

${DATASET_RULES}

Phase block rules:
- EXACTLY 7 entries: num "01" through "07", in order, titles must match:
  01 Understand the Problem
  02 Data Exploration & Cleaning
  03 Feature Selection & Preprocessing
  04 Model Selection & Training
  05 Optimization
  06 Evaluation & Validation
  07 Deployment
- Each block has 2-4 \`signals\`. Each \`quote\` must appear VERBATIM in the brief (a meaningful clause, not a single word). Each \`implication\` is a concrete, actionable decision — a specific metric, encoder, threshold, audit, monitor — not a generic platitude.
- \`intro\` is a single short sentence framing what the phase extracts from the brief.`;

export function buildBriefReadingUserMessage(type: ScenarioType, complexity: Complexity): string {
  const resolvedType =
    type === 'random' ? (Math.random() < 0.5 ? 'classification' : 'regression') : type;
  const resolvedComplexity =
    complexity === 'random'
      ? (['easy', 'medium', 'hard'] as const)[Math.floor(Math.random() * 3)]
      : complexity;

  return `Generate a fresh brief + matching dataset + 7-phase signal extraction.

Type: ${TYPE_DIRECTIVES[resolvedType]}

Complexity: ${COMPLEXITY_DIRECTIVES[resolvedComplexity]}

Pick a fresh industry / domain (NOT restaurants, fraud, churn, or housing). Every quoted phrase in the phases array must appear verbatim in your generated brief.`;
}

export const BRIEF_READING_RESPONSE_SCHEMA = {
  type: 'object',
  required: ['brief', 'dataset', 'phases'],
  properties: {
    brief: { type: 'string', minLength: 200 },
    dataset: DATASET_JSON_SCHEMA,
    phases: {
      type: 'array',
      minItems: 7,
      maxItems: 7,
      items: {
        type: 'object',
        required: ['num', 'title', 'intro', 'signals'],
        properties: {
          num: { type: 'string', enum: ['01', '02', '03', '04', '05', '06', '07'] },
          title: { type: 'string' },
          intro: { type: 'string' },
          signals: {
            type: 'array',
            minItems: 2,
            maxItems: 4,
            items: {
              type: 'object',
              required: ['quote', 'implication'],
              properties: {
                quote: { type: 'string' },
                implication: { type: 'string' },
              },
              additionalProperties: false,
            },
          },
        },
        additionalProperties: false,
      },
    },
  },
  additionalProperties: false,
};
