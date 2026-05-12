import Link from 'next/link';
import { Layers, ArrowUpRight, ArrowRight } from 'lucide-react';
import { listLearnTracks } from '@/lib/content';

export const metadata = {
  title: 'Learn | Classical ML From Scratch',
  description:
    'The complete classical ML pipeline — seven phases that take you from problem brief to deployed model.',
};

export default function LearnPage() {
  const tracks = listLearnTracks();
  const phasesTrack = tracks.find((t) => t.id === 'phases');

  return (
    <>
      {/* Hero */}
      <section className="py-12 md:py-16 border-b border-gray-200 dark:border-gray-800">
        <div className="max-w-4xl mx-auto px-4 md:px-6">
          <div className="text-[10px] font-mono tracking-[0.3em] uppercase text-gray-400 mb-2">
            STRUCTURED PATH / FOR COMPLETE BEGINNERS
          </div>
          <h1 className="font-orbitron text-3xl md:text-5xl font-bold tracking-tight mb-4">
            THE CLASSICAL ML PIPELINE
          </h1>
          <p className="text-gray-600 dark:text-gray-300 max-w-3xl text-base md:text-lg leading-relaxed">
            Seven phases that take you from problem brief to deployed model. Each phase
            has its own deep-dive page covering steps, decisions, code, common mistakes,
            and worked examples from the scenario catalog.
          </p>
        </div>
      </section>

      {/* Phase deep-dive cards */}
      {phasesTrack && (
        <section className="max-w-7xl mx-auto px-4 md:px-6 py-12">
          <div className="flex items-end justify-between mb-8 border-b border-gray-200 dark:border-gray-800 pb-4">
            <div className="flex items-center gap-3">
              <Layers className="w-5 h-5 text-gray-400" />
              <div>
                <div className="text-[10px] font-mono tracking-[0.3em] uppercase text-gray-400">
                  Deep dives
                </div>
                <h2 className="font-orbitron text-xl md:text-2xl font-bold tracking-wider">
                  PIPELINE PHASES
                </h2>
              </div>
            </div>
            <p className="hidden md:block text-xs text-gray-500 dark:text-gray-400 italic">
              {phasesTrack.subtitle}
            </p>
          </div>

          <ul className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {phasesTrack.articles.map((a) => {
              const phaseNum = a.slug.match(/phase_(\d+)/)?.[1]?.padStart(2, '0') ?? '··';
              const displayTitle = a.title.replace(/^Phase\s+\d+:\s*/i, '');
              if (!a.exists) {
                return (
                  <li key={a.slug}>
                    <div className="h-full border border-dashed border-gray-300 dark:border-gray-700 bg-gray-50/50 dark:bg-gray-900/30 px-5 py-5 opacity-60">
                      <div className="flex items-center justify-between mb-3">
                        <div className="text-[10px] font-mono tracking-[0.3em] uppercase text-gray-400">
                          Phase {phaseNum}
                        </div>
                        <div className="text-[10px] font-mono tracking-[0.2em] uppercase text-amber-600 dark:text-amber-400">
                          Soon
                        </div>
                      </div>
                      <h3 className="text-base font-semibold leading-tight mb-2 text-gray-700 dark:text-gray-300">
                        {displayTitle}
                      </h3>
                      {a.summary && (
                        <p className="text-xs text-gray-500 dark:text-gray-400 leading-relaxed line-clamp-3">
                          {a.summary}
                        </p>
                      )}
                    </div>
                  </li>
                );
              }
              return (
                <li key={a.slug}>
                  <Link
                    href={`/learn/${a.slug}`}
                    className="group flex h-full flex-col border border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-900 px-5 py-5 hover:border-emerald-600 dark:hover:border-emerald-400 hover:bg-emerald-50/50 dark:hover:bg-emerald-950/20 transition-colors"
                  >
                    <div className="flex items-center justify-between mb-3">
                      <div className="text-[10px] font-mono tracking-[0.3em] uppercase text-gray-400 group-hover:text-emerald-700 dark:group-hover:text-emerald-300 transition-colors">
                        Phase {phaseNum}
                      </div>
                      <ArrowUpRight className="w-4 h-4 text-gray-300 dark:text-gray-700 group-hover:text-emerald-600 dark:group-hover:text-emerald-400 group-hover:translate-x-0.5 group-hover:-translate-y-0.5 transition-all" />
                    </div>
                    <h3 className="text-base font-semibold leading-snug mb-2 group-hover:text-emerald-800 dark:group-hover:text-emerald-200 transition-colors">
                      {displayTitle}
                    </h3>
                    {a.summary && (
                      <p className="text-xs text-gray-500 dark:text-gray-400 leading-relaxed line-clamp-3 flex-1">
                        {a.summary}
                      </p>
                    )}
                  </Link>
                </li>
              );
            })}
          </ul>

          {/* CTA — linear walkthrough entry point */}
          <div className="mt-10 flex justify-center">
            <Link
              href="/learn/04_phase_1_understand_problem"
              className="group inline-flex items-center gap-3 px-6 py-3 border-2 border-black dark:border-white bg-white dark:bg-gray-900 text-black dark:text-white text-xs uppercase tracking-widest font-semibold hover:bg-emerald-50 dark:hover:bg-emerald-950/40 hover:border-emerald-600 dark:hover:border-emerald-400 hover:text-emerald-800 dark:hover:text-emerald-200 transition-colors"
            >
              Start with Phase 1 — Understand the Problem
              <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
            </Link>
          </div>
        </section>
      )}
    </>
  );
}
