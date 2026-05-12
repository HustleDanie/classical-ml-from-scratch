/**
 * System prompts for the three Claude API endpoints.
 *
 *   /api/practice/brief         — generate a fresh ML business brief
 *   /api/practice/solution      — generate the 17-step expert solution
 *                                  (+ critique if the user wrote an attempt)
 *   /api/learn/brief-reading    — generate a brief + a 7-phase signal
 *                                  extraction in one document
 *
 * Each system prompt is stable across calls so the Anthropic prompt cache
 * (5-min TTL) cuts cost ~90% on the prefix for repeat traffic.
 */

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
 * BRIEF generation
 * --------------------------------------------------------------------- */

export const BRIEF_SYSTEM_PROMPT = `You generate realistic machine-learning business problem briefs for a learner.

Output is ONLY the brief — no solution, no algorithm hints, no analytical commentary. The brief is what an ML engineer would receive from a stakeholder on day one.

Format the brief in markdown EXACTLY like this template (the bracketed parts are placeholders, not literal):

# [Scenario title — descriptive, e.g. "Hospital Readmission Risk Scoring"]

> **Complexity:** [one-line callout that names the 2-3 hardest constraints — e.g. "Moderate imbalance (~9%), regulated under HIPAA, real-time at discharge, mixed types with informative missingness."]

---

## The Brief

[2-4 paragraphs from a stakeholder's voice. Include:
- The business / domain (specific: "a regional grocery chain", not "a company")
- The exact prediction target and what business decision it drives
- Data shape: row count, feature count, target distribution
- Hard constraints: latency budget, regulatory regime, cost asymmetry
- What "good" looks like: the primary metric AND the threshold a stakeholder cares about]

[Optional final paragraph: a "this is harder than it sounds because…" line that sets up the trap learners typically fall into.]

---

Rules:
- DO NOT include any solution, algorithm recommendation, pipeline steps, or "what to do" advice.
- DO NOT include section headers beyond "# Title" and "## The Brief".
- Use realistic numbers (plausible row counts, percentages, dollar amounts).
- Voice: a real stakeholder — concrete, slightly informal, with implicit urgency.
- 200-400 words total.`;

export function buildBriefUserMessage(type: ScenarioType, complexity: Complexity): string {
  const resolvedType =
    type === 'random' ? (Math.random() < 0.5 ? 'classification' : 'regression') : type;
  const resolvedComplexity =
    complexity === 'random'
      ? (['easy', 'medium', 'hard'] as const)[Math.floor(Math.random() * 3)]
      : complexity;

  return `Generate a fresh problem brief.

Type: ${TYPE_DIRECTIVES[resolvedType]}

Complexity: ${COMPLEXITY_DIRECTIVES[resolvedComplexity]}

Pick an industry / domain at random — vary it across calls (don't default to fraud, churn, or housing). Make the brief specific and concrete.`;
}

/* -----------------------------------------------------------------------
 * SOLUTION generation (17-step walkthrough + critique)
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
 * BRIEF READING — brief + 7-phase signal extraction
 * --------------------------------------------------------------------- */

export const BRIEF_READING_SYSTEM_PROMPT = `You generate a teaching artefact: a realistic ML problem brief, followed by a per-phase dissection that pulls out the exact phrases that drive decisions in each of the 7 pipeline phases.

Output in markdown EXACTLY in this shape:

# [Scenario title]

> **Complexity:** [2-3 hardest constraints]

---

## The Brief

[2-4 paragraphs from a stakeholder's voice. 200-400 words. Same standard as a normal brief.]

---

## Signal Extraction

### Phase 1 — Understand the Problem

> *"[exact verbatim phrase from the brief above]"*
> → [decision this phrase forces in Phase 1: metric, problem type, regulatory frame, cost structure]

> *"[another verbatim phrase]"*
> → [the decision]

(2-4 quoted phrases per phase. Each phrase MUST appear verbatim in the brief above.)

### Phase 2 — Data Exploration & Cleaning
(same pattern)

### Phase 3 — Feature Selection & Preprocessing
(same pattern)

### Phase 4 — Model Selection & Training
(same pattern)

### Phase 5 — Optimization
(same pattern)

### Phase 6 — Evaluation & Validation
(same pattern)

### Phase 7 — Deployment
(same pattern)

Rules:
- Every quoted phrase must be VERBATIM from the brief you wrote.
- The implication arrow ("→") must be a concrete decision a learner can act on — specific encoders, specific metrics, specific thresholds — not generic platitudes.
- Don't reference data / columns that aren't in the brief.
- No section headings beyond what's listed above.`;

export function buildBriefReadingUserMessage(type: ScenarioType, complexity: Complexity): string {
  const resolvedType =
    type === 'random' ? (Math.random() < 0.5 ? 'classification' : 'regression') : type;
  const resolvedComplexity =
    complexity === 'random'
      ? (['easy', 'medium', 'hard'] as const)[Math.floor(Math.random() * 3)]
      : complexity;

  return `Generate a fresh brief + 7-phase signal extraction.

Type: ${TYPE_DIRECTIVES[resolvedType]}

Complexity: ${COMPLEXITY_DIRECTIVES[resolvedComplexity]}

Pick an industry / domain at random (NOT restaurants, NOT fraud, NOT churn — those are over-used). Make every quoted phrase appear verbatim in your own brief.`;
}
