import Link from 'next/link';
import { Layers } from 'lucide-react';
import {
  listLearnTracks,
  readLearnArticle,
} from '@/lib/content';
import { MarkdownRenderer } from '@/components/MarkdownRenderer';
import { HeroGrid } from '@/components/HeroGrid';

export const metadata = {
  title: 'Learn | Classical ML From Scratch',
  description:
    'The complete classical ML pipeline — seven phases, every step. The end-to-end framework every expert scenario follows.',
};

export default function LearnPage() {
  const tracks = listLearnTracks();
  const phasesTrack = tracks.find((t) => t.id === 'phases');
  const pipelineBody = readLearnArticle('03_ml_pipeline');

  return (
    <>
      {/* Hero */}
      <section className="relative py-12 md:py-16 overflow-hidden border-b border-gray-200 dark:border-gray-800">
        <HeroGrid />
        <div className="relative max-w-4xl mx-auto px-4 md:px-6">
          <div className="text-[10px] font-mono tracking-[0.3em] uppercase text-gray-400 mb-2">
            STRUCTURED PATH / FOR COMPLETE BEGINNERS
          </div>
          <h1 className="font-orbitron text-3xl md:text-5xl font-bold tracking-tight mb-4">
            THE CLASSICAL ML PIPELINE
          </h1>
          <p className="text-gray-600 dark:text-gray-300 max-w-3xl text-base md:text-lg leading-relaxed">
            The full end-to-end pipeline to follow every time you&apos;re given a classical
            ML task. Read this overview first, then walk through each of the seven phases
            in depth using the deep-dive pages below.
          </p>
        </div>
      </section>

      {/* Phase deep-dive nav */}
      {phasesTrack && (
        <section className="max-w-7xl mx-auto px-4 md:px-6 py-10">
          <div className="flex items-end justify-between mb-6 border-b border-gray-200 dark:border-gray-800 pb-3">
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

          <ul className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
            {phasesTrack.articles.map((a) => {
              const num = a.slug.match(/^(\d+)_/)?.[1] ?? '··';
              if (!a.exists) {
                return (
                  <li key={a.slug}>
                    <div className="block h-full border border-dashed border-gray-300 dark:border-gray-700 bg-gray-50 dark:bg-gray-900/40 px-4 py-3 opacity-60">
                      <div className="flex items-center justify-between mb-1">
                        <div className="text-[10px] font-mono tracking-[0.3em] uppercase text-gray-400">
                          {num}
                        </div>
                        <div className="text-[10px] font-mono tracking-[0.2em] uppercase text-amber-600 dark:text-amber-400">
                          Coming soon
                        </div>
                      </div>
                      <div className="text-sm font-semibold leading-tight mb-1 text-gray-700 dark:text-gray-300">
                        {a.title}
                      </div>
                      {a.summary && (
                        <p className="text-xs text-gray-500 dark:text-gray-400 leading-relaxed line-clamp-2">
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
                    className="group block h-full border border-gray-200 dark:border-gray-800 hover:border-black dark:hover:border-white bg-white dark:bg-gray-900 px-4 py-3 transition-colors"
                  >
                    <div className="text-[10px] font-mono tracking-[0.3em] uppercase text-gray-400 mb-1">
                      {num}
                    </div>
                    <div className="text-sm font-semibold leading-tight mb-1">
                      {a.title}
                    </div>
                    {a.summary && (
                      <p className="text-xs text-gray-500 dark:text-gray-400 leading-relaxed line-clamp-2">
                        {a.summary}
                      </p>
                    )}
                  </Link>
                </li>
              );
            })}
          </ul>
        </section>
      )}

      {/* The pipeline guide content itself */}
      <section className="max-w-4xl mx-auto px-4 md:px-6 py-10">
        <article className="prose prose-zinc dark:prose-invert max-w-none">
          {pipelineBody ? (
            <MarkdownRenderer source={pipelineBody} />
          ) : (
            <p className="text-gray-500 italic">Pipeline guide content not found.</p>
          )}
        </article>
      </section>
    </>
  );
}
