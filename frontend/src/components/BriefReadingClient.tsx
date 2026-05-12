'use client';

import { useRef, useState } from 'react';
import { Loader2, RefreshCcw, Sparkles } from 'lucide-react';
import { MarkdownRenderer } from './MarkdownRenderer';

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

/**
 * Interactive island on /learn/brief-reading. Default view = the static
 * worked example rendered by the server component. Clicking "Generate New"
 * shows a fresh AI-generated brief + 7-phase dissection inline (replaces
 * the static example until reset).
 */
export function BriefReadingClient() {
  const [type, setType] = useState<ScenarioType>('random');
  const [complexity, setComplexity] = useState<Complexity>('medium');
  const [body, setBody] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const ctrlRef = useRef<AbortController | null>(null);

  async function handleGenerate() {
    setError(null);
    setBody('');
    setLoading(true);
    ctrlRef.current?.abort();
    const ctrl = new AbortController();
    ctrlRef.current = ctrl;
    const r = await streamInto(
      '/api/learn/brief-reading',
      { type, complexity },
      (acc) => setBody(acc),
      ctrl.signal,
    );
    setLoading(false);
    if (!r.ok && r.errorMessage) setError(r.errorMessage);
  }

  function handleReset() {
    ctrlRef.current?.abort();
    setBody('');
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
            Spin up a fresh AI-generated brief and 7-phase dissection. The default
            example below stays in place until you do.
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
              {body ? 'Generate Another' : 'Generate New Worked Example'}
            </button>

            {(body || loading) && (
              <button
                type="button"
                onClick={handleReset}
                className="inline-flex items-center gap-2 px-4 py-2 text-xs uppercase tracking-widest border border-gray-300 dark:border-gray-700 text-gray-600 dark:text-gray-400 hover:border-current transition-colors"
              >
                <RefreshCcw className="w-3.5 h-3.5" /> Show default example
              </button>
            )}
          </div>

          {error && (
            <div className="mt-4 border border-red-500/40 bg-red-500/5 px-3 py-2 text-sm text-red-600 dark:text-red-400">
              {error}
            </div>
          )}
        </div>
      </section>

      {/* AI-generated body — replaces the static default below it when present */}
      {(body || loading) && (
        <section className="max-w-4xl mx-auto px-4 md:px-6 pb-16">
          <div className="text-[10px] font-mono tracking-[0.3em] uppercase text-emerald-700 dark:text-emerald-300 mb-4">
            AI-generated · {type} · {complexity}
          </div>
          <div className="border border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-900 p-6 md:p-8">
            {body ? (
              <article className="prose-mlfs max-w-none">
                <MarkdownRenderer source={body} />
              </article>
            ) : (
              <p className="text-gray-500 italic">Generating fresh brief + dissection…</p>
            )}
          </div>
        </section>
      )}
    </>
  );
}
