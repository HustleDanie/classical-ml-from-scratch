'use client';

import { useEffect, useState } from 'react';
import { Loader2, RefreshCcw, FlaskConical, Send } from 'lucide-react';
import { MarkdownRenderer } from './MarkdownRenderer';
import { SAMPLE_BRIEF, SAMPLE_SOLUTION } from '@/lib/practice-sample';

interface PersistedState {
  brief: string;
  attempt: string;
  solution: string;
}

const STORAGE_KEY = 'practice.workspace.v2';

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

export function PracticeWorkspace() {
  const [brief, setBrief] = useState('');
  const [attempt, setAttempt] = useState('');
  const [solution, setSolution] = useState('');
  const [revealing, setRevealing] = useState(false);
  const [hydrated, setHydrated] = useState(false);

  // Hydrate from localStorage on mount.
  /* eslint-disable react-hooks/set-state-in-effect */
  useEffect(() => {
    const saved = loadPersisted();
    if (saved) {
      setBrief(saved.brief);
      setAttempt(saved.attempt);
      setSolution(saved.solution);
    }
    setHydrated(true);
  }, []);
  /* eslint-enable react-hooks/set-state-in-effect */

  // Persist after hydration.
  useEffect(() => {
    if (!hydrated) return;
    savePersisted({ brief, attempt, solution });
  }, [hydrated, brief, attempt, solution]);

  function handleLoadSample() {
    setBrief(SAMPLE_BRIEF);
    setAttempt('');
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
    setAttempt('');
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
          Click below to load a sample classical ML business brief. Walk through it,
          write your approach, then reveal the 17-step expert solution.
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

          {(brief || attempt || solution) && (
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

      {/* Attempt panel */}
      {brief && (
        <section className="border border-gray-200 dark:border-gray-800 p-5 bg-white dark:bg-gray-900">
          <div className="text-[10px] font-mono tracking-[0.3em] uppercase text-gray-400 mb-3">
            Step 03 — Your Approach
          </div>
          <p className="text-sm text-gray-600 dark:text-gray-300 mb-3">
            Before you read the model answer, write down how you&apos;d tackle this.
            Outline the problem type, the metric you&apos;d optimise, what you&apos;d
            check first, the model family you&apos;d try, and the trap you&apos;d
            watch for. The thinking is the point — even a rough sketch beats none.
          </p>
          <textarea
            value={attempt}
            onChange={(e) => setAttempt(e.target.value)}
            placeholder="My approach…"
            rows={10}
            className="w-full px-3 py-2 border border-gray-300 dark:border-gray-700 bg-transparent font-mono text-sm focus:outline-none focus:border-black dark:focus:border-white transition-colors"
          />
          <div className="mt-4 flex flex-wrap gap-3">
            <button
              type="button"
              disabled={revealing}
              onClick={handleRevealSolution}
              className="inline-flex items-center gap-2 px-4 py-2 text-xs uppercase tracking-widest border border-black dark:border-white bg-black dark:bg-white text-white dark:text-black hover:opacity-80 transition-opacity disabled:opacity-50"
            >
              {revealing ? (
                <Loader2 className="w-3.5 h-3.5 animate-spin" />
              ) : (
                <Send className="w-3.5 h-3.5" />
              )}
              Reveal Expert Solution
            </button>
            <span className="self-center text-xs text-gray-500 dark:text-gray-400">
              {attempt.length === 0
                ? 'Revealing without writing is allowed — but you learn more by writing first.'
                : `${attempt.split(/\s+/).filter(Boolean).length} words`}
            </span>
          </div>
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
