'use client';

import { useRef, useState } from 'react';
import { Loader2, RefreshCcw, Sparkles, ArrowRight } from 'lucide-react';
import { MarkdownRenderer } from './MarkdownRenderer';
import { GeneratedDatasetPreview } from './GeneratedDatasetPreview';
import type { GeneratedDataset } from '@/lib/dataset';
import Link from 'next/link';

type ScenarioType = 'classification' | 'regression' | 'random';
type Complexity = 'easy' | 'medium' | 'hard' | 'random';

const TYPE_OPTIONS: { value: ScenarioType; label: string }[] = [
  { value: 'classification', label: 'Classification' },
  { value: 'regression', label: 'Regression' },
  { value: 'random', label: 'Random' },
];

const COMPLEXITY_OPTIONS: { value: Complexity; label: string }[] = [
  { value: 'easy', label: 'Easy' },
  { value: 'medium', label: 'Medium' },
  { value: 'hard', label: 'Hard' },
  { value: 'random', label: 'Random' },
];

interface PhaseBlock {
  num: string;
  title: string;
  intro: string;
  signals: { quote: string; implication: string }[];
}

interface BriefReadingResponse {
  brief: string;
  dataset: GeneratedDataset;
  phases: PhaseBlock[];
}

const PHASE_SLUG: Record<string, string> = {
  '01': '04_phase_1_understand_problem',
  '02': '05_phase_2_data_exploration_cleaning',
  '03': '06_phase_3_feature_selection_preprocessing',
  '04': '07_phase_4_model_selection_training',
  '05': '08_phase_5_optimization',
  '06': '09_phase_6_evaluation_validation',
  '07': '10_phase_7_deployment',
};

async function postJson<T>(
  url: string,
  body: unknown,
  signal?: AbortSignal,
): Promise<{ ok: boolean; data?: T; errorMessage?: string }> {
  try {
    const res = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
      signal,
    });
    const data = await res.json().catch(() => null);
    if (!res.ok) {
      return {
        ok: false,
        errorMessage: data?.error
          ? String(data.error)
          : `Request failed (${res.status}).`,
      };
    }
    return { ok: true, data: data as T };
  } catch (err) {
    return {
      ok: false,
      errorMessage: err instanceof Error ? err.message : 'Network error.',
    };
  }
}

/**
 * Interactive island on /learn/brief-reading. Default = the hand-crafted
 * worked example rendered below. Clicking Generate triggers /api/learn/
 * brief-reading which returns { brief, dataset, phases } in a single
 * structured-output call. The generated artefact replaces the default
 * until reset.
 */
export function BriefReadingClient() {
  const [type, setType] = useState<ScenarioType>('random');
  const [complexity, setComplexity] = useState<Complexity>('medium');
  const [response, setResponse] = useState<BriefReadingResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const ctrlRef = useRef<AbortController | null>(null);

  async function handleGenerate() {
    setError(null);
    setResponse(null);
    setLoading(true);
    ctrlRef.current?.abort();
    const ctrl = new AbortController();
    ctrlRef.current = ctrl;
    const r = await postJson<BriefReadingResponse>(
      '/api/learn/brief-reading',
      { type, complexity },
      ctrl.signal,
    );
    setLoading(false);
    if (r.ok && r.data) {
      setResponse(r.data);
    } else if (r.errorMessage) {
      setError(r.errorMessage);
    }
  }

  function handleReset() {
    ctrlRef.current?.abort();
    setResponse(null);
    setError(null);
  }

  return (
    <>
      {/* Controls */}
      <section className="max-w-4xl mx-auto px-4 md:px-6 pb-10">
        <div className="text-[10px] font-mono tracking-[0.3em] uppercase text-gray-400 mb-4">
          Generate a new worked example
        </div>
        <div className="border border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-900 p-5">
          <p className="text-sm text-gray-600 dark:text-gray-300 mb-5">
            Generate a fresh brief, its matching dataset, and a 7-phase dissection
            that pulls verbatim phrases from the brief and maps each to a concrete
            decision.
          </p>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-5 mb-5">
            <div>
              <div className="text-xs uppercase tracking-widest text-gray-500 dark:text-gray-400 mb-2">
                Type
              </div>
              <div className="flex flex-wrap gap-2">
                {TYPE_OPTIONS.map((opt) => {
                  const active = type === opt.value;
                  return (
                    <button
                      key={opt.value}
                      type="button"
                      onClick={() => setType(opt.value)}
                      className={`px-3 py-1.5 text-xs uppercase tracking-widest border transition-colors ${
                        active
                          ? 'border-emerald-600 dark:border-emerald-400 text-emerald-700 dark:text-emerald-300'
                          : 'border-gray-300 dark:border-gray-700 text-gray-500 hover:border-current'
                      }`}
                    >
                      {opt.label}
                    </button>
                  );
                })}
              </div>
            </div>

            <div>
              <div className="text-xs uppercase tracking-widest text-gray-500 dark:text-gray-400 mb-2">
                Complexity
              </div>
              <div className="flex flex-wrap gap-2">
                {COMPLEXITY_OPTIONS.map((opt) => {
                  const active = complexity === opt.value;
                  return (
                    <button
                      key={opt.value}
                      type="button"
                      onClick={() => setComplexity(opt.value)}
                      className={`px-3 py-1.5 text-xs uppercase tracking-widest border transition-colors ${
                        active
                          ? 'border-black dark:border-white text-black dark:text-white'
                          : 'border-gray-300 dark:border-gray-700 text-gray-500 hover:border-current'
                      }`}
                    >
                      {opt.label}
                    </button>
                  );
                })}
              </div>
            </div>
          </div>

          <div className="flex flex-wrap gap-3">
            <button
              type="button"
              disabled={loading}
              onClick={handleGenerate}
              className="inline-flex items-center gap-2 px-4 py-2 text-xs uppercase tracking-widest border border-black dark:border-white bg-black dark:bg-white text-white dark:text-black hover:bg-emerald-600 hover:border-emerald-600 dark:hover:bg-emerald-500 dark:hover:border-emerald-500 transition-colors disabled:opacity-50"
            >
              {loading ? (
                <Loader2 className="w-3.5 h-3.5 animate-spin" />
              ) : (
                <Sparkles className="w-3.5 h-3.5" />
              )}
              {response ? 'Generate Another' : 'Generate New Worked Example'}
            </button>

            {(response || loading) && (
              <button
                type="button"
                onClick={handleReset}
                className="inline-flex items-center gap-2 px-4 py-2 text-xs uppercase tracking-widest border border-gray-300 dark:border-gray-700 text-gray-600 dark:text-gray-400 hover:border-current transition-colors"
              >
                <RefreshCcw className="w-3.5 h-3.5" /> Reset
              </button>
            )}
          </div>

          {error && (
            <div className="mt-4 border border-red-500/40 bg-red-500/5 px-3 py-2 text-sm text-red-600 dark:text-red-400">
              {error}
            </div>
          )}

          {loading && !response && (
            <div className="mt-4 text-xs text-gray-500 dark:text-gray-400 italic">
              Generating brief + dataset + 7-phase dissection… this can take 60–120
              seconds with adaptive thinking.
            </div>
          )}
        </div>
      </section>

      {/* AI-generated artefact */}
      {response && (
        <>
          <section className="max-w-4xl mx-auto px-4 md:px-6 pb-2">
            <div className="text-[10px] font-mono tracking-[0.3em] uppercase text-emerald-700 dark:text-emerald-300 mb-1">
              AI-generated · {type} · {complexity}
            </div>
          </section>

          {/* Brief */}
          <section className="max-w-4xl mx-auto px-4 md:px-6 pb-10">
            <div className="text-[10px] font-mono tracking-[0.3em] uppercase text-gray-400 mb-4">
              The Brief
            </div>
            <div className="border border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-900 p-6 md:p-8">
              <div className="prose-mlfs max-w-none">
                <MarkdownRenderer source={response.brief} />
              </div>
            </div>
          </section>

          {/* Dataset */}
          <section className="max-w-4xl mx-auto px-4 md:px-6 pb-12">
            <div className="text-[10px] font-mono tracking-[0.3em] uppercase text-gray-400 mb-4">
              The Dataset
            </div>
            <div className="border border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-900 p-6 md:p-8">
              <GeneratedDatasetPreview dataset={response.dataset} framed={false} />
            </div>
          </section>

          {/* Per-phase signal extraction */}
          <section className="max-w-4xl mx-auto px-4 md:px-6 pb-16 md:pb-24">
            <div className="text-[10px] font-mono tracking-[0.3em] uppercase text-gray-400 mb-4">
              Signal extraction
            </div>

            <ol className="space-y-12 md:space-y-16">
              {response.phases.map((p) => {
                const slug = PHASE_SLUG[p.num];
                return (
                  <li key={p.num}>
                    <div className="grid grid-cols-[auto,1fr] gap-x-6 md:gap-x-10 items-baseline mb-4">
                      <span className="font-mono text-sm tracking-[0.2em] text-gray-400">
                        {p.num}
                      </span>
                      <div>
                        {slug ? (
                          <Link
                            href={`/learn/${slug}`}
                            className="group inline-flex items-baseline gap-3 text-2xl md:text-3xl font-[family-name:var(--font-plex-serif)] hover:text-emerald-800 dark:hover:text-emerald-200 transition-colors"
                          >
                            {p.title}
                            <ArrowRight className="w-4 h-4 text-gray-300 dark:text-gray-700 group-hover:text-emerald-600 dark:group-hover:text-emerald-400 group-hover:translate-x-0.5 transition-all" />
                          </Link>
                        ) : (
                          <span className="text-2xl md:text-3xl font-[family-name:var(--font-plex-serif)]">
                            {p.title}
                          </span>
                        )}
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
                );
              })}
            </ol>
          </section>
        </>
      )}
    </>
  );
}
