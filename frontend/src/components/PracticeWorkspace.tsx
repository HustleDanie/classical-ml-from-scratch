'use client';

import { useEffect, useState } from 'react';
import { Loader2, RefreshCcw, FlaskConical, NotebookPen, Send, Download, Database } from 'lucide-react';
import Link from 'next/link';
import { MarkdownRenderer } from './MarkdownRenderer';
import {
  SAMPLE_BRIEF,
  SAMPLE_SOLUTION,
  SAMPLE_DATASET_PATH,
  SAMPLE_DATASET_COLUMNS,
  SAMPLE_DATASET_PREVIEW,
  SAMPLE_DATASET_META,
} from '@/lib/practice-sample';

const BASE_PATH = process.env.NEXT_PUBLIC_BASE_PATH ?? '';

interface PersistedState {
  brief: string;
  solution: string;
}

const STORAGE_KEY = 'practice.workspace.v3';

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

const PHASE_LIST: { num: string; title: string }[] = [
  { num: '01', title: 'Understand the Problem' },
  { num: '02', title: 'Data Exploration & Cleaning' },
  { num: '03', title: 'Feature Selection & Preprocessing' },
  { num: '04', title: 'Model Selection & Training' },
  { num: '05', title: 'Optimization' },
  { num: '06', title: 'Evaluation & Validation' },
  { num: '07', title: 'Deployment' },
];

export function PracticeWorkspace() {
  const [brief, setBrief] = useState('');
  const [solution, setSolution] = useState('');
  const [revealing, setRevealing] = useState(false);
  const [hydrated, setHydrated] = useState(false);

  /* eslint-disable react-hooks/set-state-in-effect */
  useEffect(() => {
    const saved = loadPersisted();
    if (saved) {
      setBrief(saved.brief);
      setSolution(saved.solution);
    }
    setHydrated(true);
  }, []);
  /* eslint-enable react-hooks/set-state-in-effect */

  useEffect(() => {
    if (!hydrated) return;
    savePersisted({ brief, solution });
  }, [hydrated, brief, solution]);

  function handleLoadSample() {
    setBrief(SAMPLE_BRIEF);
    setSolution('');
  }

  async function handleRevealSolution() {
    if (!brief) return;
    setSolution('');
    setRevealing(true);
    const chars = SAMPLE_SOLUTION;
    const chunkSize = 80;
    for (let i = 0; i < chars.length; i += chunkSize) {
      setSolution(chars.slice(0, i + chunkSize));
      await new Promise((r) => setTimeout(r, 12));
    }
    setRevealing(false);
  }

  function handleReset() {
    setBrief('');
    setSolution('');
    if (typeof window !== 'undefined') {
      window.localStorage.removeItem(STORAGE_KEY);
    }
  }

  return (
    <div className="space-y-10">
      {/* Controls */}
      <section className="border border-gray-200 dark:border-gray-800 p-5 bg-white dark:bg-gray-900">
        <div className="text-[10px] font-mono tracking-[0.3em] uppercase text-gray-400 mb-3">
          Step 01 — Load the brief
        </div>
        <p className="text-sm text-gray-600 dark:text-gray-300 mb-4">
          Click below to load a sample classical ML business brief. Read it, work through
          the seven phases on paper, then reveal the expert solution to compare.
        </p>

        <div className="flex flex-wrap gap-3">
          <button
            type="button"
            onClick={handleLoadSample}
            className="inline-flex items-center gap-2 px-4 py-2 text-xs uppercase tracking-widest border border-amber-500/60 bg-amber-500/10 text-amber-700 dark:text-amber-300 hover:bg-amber-500/20 transition-colors"
          >
            <FlaskConical className="w-3.5 h-3.5" />
            {brief ? 'Reload Sample Brief' : 'Load Sample Brief'}
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

      {/* Brief panel */}
      {brief && (
        <section className="border border-gray-200 dark:border-gray-800 p-5 bg-white dark:bg-gray-900">
          <div className="text-[10px] font-mono tracking-[0.3em] uppercase text-gray-400 mb-3">
            Step 02 — The Brief
          </div>
          <article className="prose prose-zinc dark:prose-invert max-w-none">
            <MarkdownRenderer source={brief} />
          </article>
        </section>
      )}

      {/* Dataset — download + preview */}
      {brief && (
        <section className="border border-gray-200 dark:border-gray-800 p-5 bg-white dark:bg-gray-900">
          <div className="text-[10px] font-mono tracking-[0.3em] uppercase text-gray-400 mb-3">
            Step 02b — The Dataset
          </div>

          <div className="flex items-start gap-3 mb-4">
            <Database className="w-5 h-5 text-emerald-600 dark:text-emerald-400 flex-shrink-0 mt-0.5" />
            <p className="text-sm text-gray-700 dark:text-gray-200 leading-relaxed">
              {SAMPLE_DATASET_META.description}
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-3 mb-4 text-xs">
            <a
              href={`${BASE_PATH}${SAMPLE_DATASET_PATH}`}
              download
              className="inline-flex items-center gap-2 px-4 py-2 uppercase tracking-widest border border-black dark:border-white bg-white dark:bg-gray-900 text-black dark:text-white font-semibold hover:bg-emerald-50 dark:hover:bg-emerald-950/40 hover:border-emerald-600 dark:hover:border-emerald-400 hover:text-emerald-800 dark:hover:text-emerald-200 transition-colors"
            >
              <Download className="w-3.5 h-3.5" />
              Download CSV
            </a>
            <span className="text-gray-500 dark:text-gray-400 font-mono tracking-widest uppercase text-[10px]">
              {SAMPLE_DATASET_META.rows} rows · critical {SAMPLE_DATASET_META.criticalRate} · renovation_disclosed missing {SAMPLE_DATASET_META.missingRenovationRate}
            </span>
          </div>

          <div className="overflow-x-auto border border-gray-200 dark:border-gray-800">
            <table className="min-w-full text-[11px] font-mono">
              <thead>
                <tr className="border-b border-gray-200 dark:border-gray-800 bg-gray-50 dark:bg-gray-900/60">
                  {SAMPLE_DATASET_COLUMNS.map((col) => (
                    <th
                      key={col}
                      className="px-3 py-2 text-left tracking-widest uppercase text-[9px] text-gray-500 dark:text-gray-400 whitespace-nowrap"
                    >
                      {col}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {SAMPLE_DATASET_PREVIEW.map((row, i) => (
                  <tr
                    key={i}
                    className={`${
                      i < SAMPLE_DATASET_PREVIEW.length - 1
                        ? 'border-b border-gray-100 dark:border-gray-800/60'
                        : ''
                    } ${
                      row.result === 'critical'
                        ? 'bg-amber-50/40 dark:bg-amber-950/10'
                        : ''
                    }`}
                  >
                    {SAMPLE_DATASET_COLUMNS.map((col) => (
                      <td
                        key={col}
                        className="px-3 py-2 text-gray-700 dark:text-gray-300 whitespace-nowrap"
                      >
                        {row[col] || (
                          <span className="text-gray-300 dark:text-gray-700 italic">∅</span>
                        )}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="mt-3 text-[10px] text-gray-500 dark:text-gray-400 font-mono">
            Showing 6 of {SAMPLE_DATASET_META.rows} rows. Critical-violation rows highlighted.
          </p>
        </section>
      )}

      {/* Pen-and-paper prompt */}
      {brief && (
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
                  href={`/learn/${{
                    '01': '04_phase_1_understand_problem',
                    '02': '05_phase_2_data_exploration_cleaning',
                    '03': '06_phase_3_feature_selection_preprocessing',
                    '04': '07_phase_4_model_selection_training',
                    '05': '08_phase_5_optimization',
                    '06': '09_phase_6_evaluation_validation',
                    '07': '10_phase_7_deployment',
                  }[p.num]}`}
                  className="text-gray-700 dark:text-gray-200 hover:text-emerald-700 dark:hover:text-emerald-300 transition-colors"
                >
                  {p.title}
                </Link>
              </li>
            ))}
          </ol>

          <button
            type="button"
            disabled={revealing}
            onClick={handleRevealSolution}
            className="inline-flex items-center gap-2 px-4 py-2 text-xs uppercase tracking-widest border border-black dark:border-white bg-black dark:bg-white text-white dark:text-black hover:bg-emerald-600 hover:border-emerald-600 dark:hover:bg-emerald-500 dark:hover:border-emerald-500 transition-colors disabled:opacity-50"
          >
            {revealing ? (
              <Loader2 className="w-3.5 h-3.5 animate-spin" />
            ) : (
              <Send className="w-3.5 h-3.5" />
            )}
            Reveal Expert Solution
          </button>
        </section>
      )}

      {/* Solution panel */}
      {(solution || revealing) && (
        <section className="border border-gray-200 dark:border-gray-800 p-5 bg-white dark:bg-gray-900">
          <div className="text-[10px] font-mono tracking-[0.3em] uppercase text-gray-400 mb-3">
            Step 04 — Expert Solution
          </div>
          <article className="prose prose-zinc dark:prose-invert max-w-none">
            {solution ? (
              <MarkdownRenderer source={solution} />
            ) : (
              <p className="text-gray-500 italic">Revealing solution…</p>
            )}
          </article>
        </section>
      )}
    </div>
  );
}
