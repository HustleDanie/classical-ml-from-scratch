'use client';

import { useEffect, useRef, useState } from 'react';
import {
  Loader2,
  RefreshCcw,
  NotebookPen,
  Send,
  Sparkles,
} from 'lucide-react';
import Link from 'next/link';
import { MarkdownRenderer } from './MarkdownRenderer';
import { GeneratedDatasetPreview } from './GeneratedDatasetPreview';
import type { GeneratedDataset } from '@/lib/dataset';

type ScenarioType = 'classification' | 'regression' | 'random';
type Complexity = 'easy' | 'medium' | 'hard' | 'random';

interface PersistedState {
  brief: string;
  solution: string;
  dataset: GeneratedDataset | null;
}

const STORAGE_KEY = 'practice.workspace.v6';

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

const PHASE_LIST = [
  { num: '01', title: 'Understand the Problem', slug: '04_phase_1_understand_problem' },
  { num: '02', title: 'Data Exploration & Cleaning', slug: '05_phase_2_data_exploration_cleaning' },
  { num: '03', title: 'Feature Selection & Preprocessing', slug: '06_phase_3_feature_selection_preprocessing' },
  { num: '04', title: 'Model Selection & Training', slug: '07_phase_4_model_selection_training' },
  { num: '05', title: 'Optimization', slug: '08_phase_5_optimization' },
  { num: '06', title: 'Evaluation & Validation', slug: '09_phase_6_evaluation_validation' },
  { num: '07', title: 'Deployment', slug: '10_phase_7_deployment' },
];

function loadPersisted(): PersistedState | null {
  if (typeof window === 'undefined') return null;
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return null;
    return JSON.parse(raw) as PersistedState;
  } catch {
    return null;
  }
}

function savePersisted(state: PersistedState) {
  if (typeof window === 'undefined') return;
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
  } catch {
    /* ignore */
  }
}

async function streamInto(
  url: string,
  body: unknown,
  onChunk: (acc: string) => void,
  signal?: AbortSignal,
): Promise<{ ok: boolean; errorMessage?: string }> {
  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
    signal,
  });
  if (!res.ok) {
    let errorMessage = `Request failed (${res.status}).`;
    try {
      const data = await res.json();
      if (data?.error) errorMessage = String(data.error);
    } catch {
      /* ignore */
    }
    return { ok: false, errorMessage };
  }
  if (!res.body) return { ok: false, errorMessage: 'No response body.' };
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let acc = '';
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    acc += decoder.decode(value, { stream: true });
    onChunk(acc);
  }
  return { ok: true };
}

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

export function PracticeWorkspace() {
  const [type, setType] = useState<ScenarioType>('random');
  const [complexity, setComplexity] = useState<Complexity>('medium');
  const [brief, setBrief] = useState('');
  const [dataset, setDataset] = useState<GeneratedDataset | null>(null);
  const [solution, setSolution] = useState('');
  const [briefLoading, setBriefLoading] = useState(false);
  const [solutionLoading, setSolutionLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [hydrated, setHydrated] = useState(false);

  const briefAbortRef = useRef<AbortController | null>(null);
  const solutionAbortRef = useRef<AbortController | null>(null);

  /* eslint-disable react-hooks/set-state-in-effect */
  useEffect(() => {
    const saved = loadPersisted();
    if (saved) {
      setBrief(saved.brief);
      setSolution(saved.solution);
      setDataset(saved.dataset ?? null);
    }
    setHydrated(true);
  }, []);
  /* eslint-enable react-hooks/set-state-in-effect */

  useEffect(() => {
    if (!hydrated) return;
    savePersisted({ brief, solution, dataset });
  }, [hydrated, brief, solution, dataset]);

  async function handleGenerateBrief() {
    setError(null);
    setBrief('');
    setDataset(null);
    setSolution('');
    setBriefLoading(true);
    briefAbortRef.current?.abort();
    const ctrl = new AbortController();
    briefAbortRef.current = ctrl;
    const r = await postJson<{ brief: string; dataset: GeneratedDataset }>(
      '/api/practice/brief',
      { type, complexity },
      ctrl.signal,
    );
    setBriefLoading(false);
    if (r.ok && r.data) {
      setBrief(r.data.brief);
      setDataset(r.data.dataset);
    } else if (r.errorMessage) {
      setError(r.errorMessage);
    }
  }

  async function handleRevealSolution() {
    if (!brief) return;
    setError(null);
    setSolution('');
    setSolutionLoading(true);
    solutionAbortRef.current?.abort();
    const ctrl = new AbortController();
    solutionAbortRef.current = ctrl;
    const r = await streamInto(
      '/api/practice/solution',
      { brief, userAttempt: '' },
      (acc) => setSolution(acc),
      ctrl.signal,
    );
    setSolutionLoading(false);
    if (!r.ok && r.errorMessage) setError(r.errorMessage);
  }

  function handleReset() {
    briefAbortRef.current?.abort();
    solutionAbortRef.current?.abort();
    setBrief('');
    setDataset(null);
    setSolution('');
    setError(null);
    if (typeof window !== 'undefined') window.localStorage.removeItem(STORAGE_KEY);
  }

  const briefAvailable = !!brief;

  return (
    <div className="space-y-10">
      {/* Controls */}
      <section className="border border-gray-200 dark:border-gray-800 p-5 bg-white dark:bg-gray-900">
        <div className="text-[10px] font-mono tracking-[0.3em] uppercase text-gray-400 mb-3">
          Step 01 — Load a brief
        </div>
        <p className="text-sm text-gray-600 dark:text-gray-300 mb-5">
          Generate a fresh AI brief and matching dataset. Read the brief, work through
          the seven phases on paper, then reveal the expert solution.
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
            disabled={briefLoading}
            onClick={handleGenerateBrief}
            className="inline-flex items-center gap-2 px-4 py-2 text-xs uppercase tracking-widest border border-black dark:border-white bg-black dark:bg-white text-white dark:text-black hover:bg-emerald-600 hover:border-emerald-600 dark:hover:bg-emerald-500 dark:hover:border-emerald-500 transition-colors disabled:opacity-50"
          >
            {briefLoading ? (
              <Loader2 className="w-3.5 h-3.5 animate-spin" />
            ) : (
              <Sparkles className="w-3.5 h-3.5" />
            )}
            {brief ? 'Generate New Brief' : 'Generate Brief'}
          </button>

          {(brief || solution) && (
            <button
              type="button"
              onClick={handleReset}
              className="inline-flex items-center gap-2 px-4 py-2 text-xs uppercase tracking-widest border border-gray-300 dark:border-gray-700 text-gray-600 dark:text-gray-400 hover:border-current transition-colors"
            >
              <RefreshCcw className="w-3.5 h-3.5" /> Reset
            </button>
          )}
        </div>
      </section>

      {error && (
        <div className="border border-red-500/40 bg-red-500/5 px-4 py-3 text-sm text-red-600 dark:text-red-400">
          {error}
        </div>
      )}

      {/* Brief panel */}
      {(brief || briefLoading) && (
        <section className="border border-gray-200 dark:border-gray-800 p-5 bg-white dark:bg-gray-900">
          <div className="text-[10px] font-mono tracking-[0.3em] uppercase text-gray-400 mb-3">
            Step 02 — The Brief
          </div>
          {brief ? (
            <article className="prose-mlfs max-w-none">
              <MarkdownRenderer source={brief} />
            </article>
          ) : (
            <p className="text-gray-500 italic">
              Generating brief and matching dataset… this can take 30–90 seconds.
            </p>
          )}
        </section>
      )}

      {/* AI-generated dataset matching the brief */}
      {briefAvailable && dataset && (
        <GeneratedDatasetPreview dataset={dataset} label="Step 02b — The Dataset" />
      )}

      {/* Pen-and-paper prompt */}
      {briefAvailable && (
        <section className="border border-gray-200 dark:border-gray-800 p-5 bg-white dark:bg-gray-900">
          <div className="text-[10px] font-mono tracking-[0.3em] uppercase text-gray-400 mb-3">
            Step 03 — Your Approach
          </div>

          <div className="flex items-start gap-3 mb-4">
            <NotebookPen className="w-5 h-5 text-emerald-600 dark:text-emerald-400 flex-shrink-0 mt-0.5" />
            <p className="text-sm text-gray-700 dark:text-gray-200">
              Grab a piece of paper. Work through the brief by writing one or two lines for
              each of the seven phases. The thinking is the point — even a rough sketch
              beats none.
            </p>
          </div>

          <ol className="grid grid-cols-1 md:grid-cols-2 gap-x-6 gap-y-2 mb-5 text-sm">
            {PHASE_LIST.map((p) => (
              <li key={p.num} className="flex items-baseline gap-3">
                <span className="font-mono text-[10px] tracking-[0.2em] uppercase text-gray-400 w-7 flex-shrink-0">
                  {p.num}
                </span>
                <Link
                  href={`/learn/${p.slug}`}
                  className="text-gray-700 dark:text-gray-200 hover:text-emerald-700 dark:hover:text-emerald-300 transition-colors"
                >
                  {p.title}
                </Link>
              </li>
            ))}
          </ol>

          <button
            type="button"
            disabled={solutionLoading}
            onClick={handleRevealSolution}
            className="inline-flex items-center gap-2 px-4 py-2 text-xs uppercase tracking-widest border border-black dark:border-white bg-black dark:bg-white text-white dark:text-black hover:bg-emerald-600 hover:border-emerald-600 dark:hover:bg-emerald-500 dark:hover:border-emerald-500 transition-colors disabled:opacity-50"
          >
            {solutionLoading ? (
              <Loader2 className="w-3.5 h-3.5 animate-spin" />
            ) : (
              <Send className="w-3.5 h-3.5" />
            )}
            Reveal Expert Solution
          </button>
        </section>
      )}

      {/* Solution panel */}
      {(solution || solutionLoading) && (
        <section className="border border-gray-200 dark:border-gray-800 p-5 bg-white dark:bg-gray-900">
          <div className="text-[10px] font-mono tracking-[0.3em] uppercase text-gray-400 mb-3">
            Step 04 — Expert Solution
          </div>
          {solution ? (
            <article className="prose-mlfs max-w-none">
              <MarkdownRenderer source={solution} />
            </article>
          ) : (
            <p className="text-gray-500 italic">Generating expert solution…</p>
          )}
        </section>
      )}
    </div>
  );
}
