import Link from 'next/link';
import { ArrowLeft, ArrowRight } from 'lucide-react';
import { MarkdownRenderer } from '@/components/MarkdownRenderer';
import { DatasetPreview } from '@/components/DatasetPreview';
import { SAMPLE_BRIEF } from '@/lib/practice-sample';

export const metadata = {
  title: 'Reading a Brief | Learn — Classical ML From Scratch',
  description:
    'Same brief, seven phases. The exact phrases an expert would extract from a real ML brief to drive every decision in the pipeline.',
};

const PHASE_BLOCKS: {
  num: string;
  title: string;
  slug: string;
  intro: string;
  signals: { quote: string; implication: string }[];
}[] = [
  {
    num: '01',
    title: 'Understand the Problem',
    slug: '04_phase_1_understand_problem',
    intro: 'Extract the prediction target, the metric, the cost structure, and the regulatory frame.',
    signals: [
      {
        quote:
          'predict which restaurants are likely to have a critical violation in the next 90 days',
        implication:
          'Binary classification with a 90-day horizon. "Given the restaurant\'s recent inspection record, predict whether the next visit finds a critical violation."',
      },
      {
        quote: 'The base rate of critical violations is roughly 7% of inspections',
        implication:
          'Imbalanced. Accuracy is useless. Primary metric is PR-AUC; report recall@k since the deployment ranks under a fixed budget.',
      },
      {
        quote:
          'a missed critical violation can cause an outbreak (six-figure response cost plus health harm), while an unnecessary inspection costs ~$240',
        implication:
          'Severe cost asymmetry → tune the threshold to minimise `c_fn × FN + c_fp × FP`. Report cost in $ alongside PR-AUC.',
      },
      {
        quote:
          "the model's decisions WILL be FOIA'd, so we need clear feature attributions per restaurant",
        implication:
          'Regulated. Calibrated probabilities are mandatory; per-row SHAP shipped with every dispatch list. Don\'t stack — keep one interpretable model.',
      },
    ],
  },
  {
    num: '02',
    title: 'Data Exploration & Cleaning',
    slug: '05_phase_2_data_exploration_cleaning',
    intro: 'Find the structure, the missingness pattern, the leakers, the free-text columns.',
    signals: [
      {
        quote:
          '60+ feature columns including cuisine type, seating capacity, ownership type, complaint history (free-text complaint logs we\'d need to handle), days since last inspection, prior violation history, neighborhood demographics, weather at inspection time',
        implication:
          'Mixed types: numeric (`seating_capacity`, `days_since_last_inspection`, `prior_critical_count`, `neighborhood_median_income`), low-cardinality categorical (`cuisine_type` 9 levels, `ownership_type` 2, `county` 10), high-cardinality (`inspector_id` 88 levels), and one free-text field (`complaint_text`). The free-text column needs TF-IDF + a sentiment / keyword flag in Phase 2; the demographics column gets a fairness audit in Phase 6.',
      },
      {
        quote:
          'a sparse field for "renovations or ownership change in last 12 months" that\'s only filled in for ~20% of records (and we suspect missingness is informative — owners who don\'t disclose are more likely to fail)',
        implication:
          'Open the CSV: `renovation_disclosed` is blank in **82%** of rows — matches the brief. Run `df.groupby(df["renovation_disclosed"] == "")["result"].apply(lambda s: (s == "critical").mean())` — the critical rate among non-disclosers is meaningfully higher. Keep a `renovation_missing` flag, *then* impute.',
      },
      {
        quote: 'the inspector ID',
        implication:
          '`inspector_id` has 88 unique values. Group `df.groupby("inspector_id")["result"].apply(lambda s: (s == "critical").mean())` — inspector-level critical rates vary 3× to 14× across inspectors. That spread is judgment, not restaurant signal. Drop the column before feature engineering.',
      },
      {
        quote: 'weather at inspection time',
        implication:
          'The supplied sample has no weather column — only `inspection_date`. Engineer cyclical features from the date (`month_sin`, `month_cos`, `is_summer`); outbreak-prone foods spike in heat and that shows up in the seasonality even without weather data.',
      },
    ],
  },
  {
    num: '03',
    title: 'Feature Selection & Preprocessing',
    slug: '06_phase_3_feature_selection_preprocessing',
    intro: 'Decide what survives, transform per type, split before fitting any encoder.',
    signals: [
      {
        quote: '6 years of historical data: ~210K inspection records',
        implication:
          'Time-based split, not random. Sort by `inspection_date`, train on early years, validate on the most recent year. Random splits would leak future inspections into training via the rolling-window features.',
      },
      {
        quote: 'cuisine type, ownership type, … neighborhood demographics',
        implication:
          'In the sample CSV: `cuisine_type` (9 levels) and `ownership_type` (2 levels) → `OneHotEncoder`. `county` (10 in this sample, ~hundreds in real data) → `TargetEncoder` with Bayesian smoothing. `inspector_id` → drop, do NOT target-encode (confounder).',
      },
      {
        quote: 'inspector judgment … feedback loop risk',
        implication:
          'Reserve a small random-inspection holdout each quarter; otherwise the training distribution narrows to "what inspectors chose to look at" and the model degrades silently.',
      },
    ],
  },
  {
    num: '04',
    title: 'Model Selection & Training',
    slug: '07_phase_4_model_selection_training',
    intro: 'Justify the model family from the data shape + the regulatory frame.',
    signals: [
      {
        quote: '~210K inspection records … quarterly batch scoring run … no real-time constraint',
        implication:
          'Large enough for tree boosters. No latency budget → LightGBM / XGBoost are fine. Bake-off LightGBM, XGBoost, RandomForest, calibrated LogReg.',
      },
      {
        quote: 'roughly 7% of inspections',
        implication:
          'Set `scale_pos_weight ≈ 13` (negatives/positives) on LGBM/XGB; `class_weight=\'balanced\'` on the LogReg challenger. Threshold-tune in Phase 6.',
      },
      {
        quote: 'feedback loop risk if we just train on "did the inspector find a violation"',
        implication:
          'Label shift. Either (a) reweight observations by inverse propensity of being inspected, or (b) keep the random-inspection holdout for honest evaluation. Document the choice.',
      },
    ],
  },
  {
    num: '05',
    title: 'Optimization',
    slug: '08_phase_5_optimization',
    intro: 'Tune the winner. Decide whether stacking is worth it.',
    signals: [
      {
        quote: "the model's decisions WILL be FOIA'd",
        implication:
          'Reject stacking. Stacked predictions are harder to explain per-row. Ship a single calibrated LightGBM — easier to defend in audit.',
      },
      {
        quote: 'quarterly batch scoring run',
        implication:
          'Generous compute budget. Use Optuna (TPE) with 80–120 trials over LGBM hyperparameters. Tune on training years, evaluate on the held-out recent year.',
      },
    ],
  },
  {
    num: '06',
    title: 'Evaluation & Validation',
    slug: '09_phase_6_evaluation_validation',
    intro: 'Calibrate first, then tune the threshold. Audit fairness.',
    signals: [
      {
        quote: 'the score will be FOIA\'d',
        implication:
          'Calibration is mandatory. Wrap the LGBM with `CalibratedClassifierCV(method=\'isotonic\')` on a held-out validation fold. Plot calibration curve in the report.',
      },
      {
        quote:
          'a missed critical violation can cause an outbreak (six-figure response cost plus health harm), while an unnecessary inspection costs ~$240',
        implication:
          'Cost-driven threshold: minimise `15 × FP + 5000 × FN` (or your real numbers). Report the cost-optimal threshold and the dollars-saved-vs-rule-baseline number.',
      },
      {
        quote:
          'pressure from the small-business association to make sure the score isn\'t systematically harder on independent restaurants vs. chains',
        implication:
          'Use the `ownership_type` column directly for the audit: `df.groupby("ownership_type")` and report recall@k, precision@k, and Brier per group. Alert if the recall gap > 5 pp. Don\'t post-hoc-adjust thresholds per group unless you can justify it legally.',
      },
    ],
  },
  {
    num: '07',
    title: 'Deployment',
    slug: '10_phase_7_deployment',
    intro: 'Persist the bundle. Plan retraining and monitoring.',
    signals: [
      {
        quote: 'a quarterly batch scoring run',
        implication:
          'Quarterly retraining cadence. The bundle is: model + isotonic calibrator + threshold + target-encoder lookup + meta.json with the training-data hash.',
      },
      {
        quote: 'feedback loop risk',
        implication:
          'Drift monitor: PSI on the top-5 SHAP features daily. Rolling PR-AUC on confirmed-outcome inspections weekly. Reserve 10% of capacity for random inspections to keep training data honest.',
      },
      {
        quote: 'small-business association … fairness',
        implication:
          'Monthly fairness audit job. If recall@k for either group drifts > 5 pp, page on-call and block promotion of the next retrained model.',
      },
    ],
  },
];

export default function BriefReadingPage() {
  return (
    <>
      {/* Breadcrumb */}
      <section className="border-b border-gray-200 dark:border-gray-800">
        <div className="max-w-5xl mx-auto px-4 md:px-6 py-4 flex items-center gap-2 text-[10px] font-mono tracking-[0.2em] uppercase text-gray-400">
          <Link
            href="/learn"
            className="hover:text-emerald-700 dark:hover:text-emerald-300 transition-colors"
          >
            Learn
          </Link>
          <span className="text-gray-300 dark:text-gray-700">/</span>
          <span className="text-black dark:text-white">Reading a Brief</span>
        </div>
      </section>

      {/* Editorial hero */}
      <section className="max-w-4xl mx-auto px-4 md:px-6 pt-12 md:pt-16 pb-8 md:pb-10">
        <div className="text-[10px] font-mono tracking-[0.3em] uppercase text-gray-400 mb-4">
          Same brief · Seven phases
        </div>
        <h1 className="font-[family-name:var(--font-plex-serif)] text-4xl md:text-6xl leading-[1.05] tracking-tight font-normal mb-4">
          Reading a Brief.
        </h1>
        <p className="font-[family-name:var(--font-plex-serif)] text-lg md:text-xl leading-snug text-gray-600 dark:text-gray-300 max-w-3xl">
          The same restaurant-inspection brief from the practice page, this time
          dissected. For every one of the seven phases, here are the exact phrases an
          expert pulls from the brief — and the decision each phrase forces.
        </p>
      </section>

      {/* The brief */}
      <section className="max-w-4xl mx-auto px-4 md:px-6 pb-10">
        <div className="text-[10px] font-mono tracking-[0.3em] uppercase text-gray-400 mb-4">
          The Brief
        </div>
        <div className="border border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-900 p-6 md:p-8">
          <div className="prose-mlfs max-w-none">
            <MarkdownRenderer source={SAMPLE_BRIEF} />
          </div>
        </div>
      </section>

      {/* The dataset */}
      <section className="max-w-4xl mx-auto px-4 md:px-6 pb-12">
        <div className="text-[10px] font-mono tracking-[0.3em] uppercase text-gray-400 mb-4">
          The Dataset
        </div>
        <div className="border border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-900 p-6 md:p-8">
          <DatasetPreview framed={false} />
        </div>
      </section>

      {/* Per-phase signal extraction */}
      <section className="max-w-4xl mx-auto px-4 md:px-6 pb-16 md:pb-24">
        <div className="text-[10px] font-mono tracking-[0.3em] uppercase text-gray-400 mb-4">
          Signal extraction
        </div>

        <ol className="space-y-12 md:space-y-16">
          {PHASE_BLOCKS.map((p) => (
            <li key={p.num}>
              <div className="grid grid-cols-[auto,1fr] gap-x-6 md:gap-x-10 items-baseline mb-4">
                <span className="font-mono text-sm tracking-[0.2em] text-gray-400">
                  {p.num}
                </span>
                <div>
                  <Link
                    href={`/learn/${p.slug}`}
                    className="group inline-flex items-baseline gap-3 text-2xl md:text-3xl font-[family-name:var(--font-plex-serif)] hover:text-emerald-800 dark:hover:text-emerald-200 transition-colors"
                  >
                    {p.title}
                    <ArrowRight className="w-4 h-4 text-gray-300 dark:text-gray-700 group-hover:text-emerald-600 dark:group-hover:text-emerald-400 group-hover:translate-x-0.5 transition-all" />
                  </Link>
                  <p className="mt-1 text-sm text-gray-500 dark:text-gray-400 italic font-[family-name:var(--font-plex-serif)]">
                    {p.intro}
                  </p>
                </div>
              </div>

              <ul className="ml-0 md:ml-12 space-y-5 border-l-2 border-gray-200 dark:border-gray-800 pl-5 md:pl-6">
                {p.signals.map((s, i) => (
                  <li key={i}>
                    <blockquote className="font-[family-name:var(--font-plex-serif)] italic text-gray-700 dark:text-gray-200 text-[1.05rem] leading-relaxed">
                      “{s.quote}”
                    </blockquote>
                    <p className="mt-2 text-sm text-gray-600 dark:text-gray-300 leading-relaxed">
                      <span className="font-mono text-[10px] tracking-[0.2em] uppercase text-emerald-700 dark:text-emerald-300 mr-2">
                        →
                      </span>
                      {s.implication}
                    </p>
                  </li>
                ))}
              </ul>
            </li>
          ))}
        </ol>
      </section>

      {/* Back to Learn */}
      <section className="max-w-4xl mx-auto px-4 md:px-6 pb-16">
        <Link
          href="/learn"
          className="inline-flex items-center gap-2 text-xs uppercase tracking-widest text-gray-500 hover:text-emerald-700 dark:hover:text-emerald-300 transition-colors"
        >
          <ArrowLeft className="w-3.5 h-3.5" />
          Back to Learn
        </Link>
      </section>
    </>
  );
}
