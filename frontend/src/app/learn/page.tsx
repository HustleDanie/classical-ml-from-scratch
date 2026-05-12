import Link from 'next/link';
import { ArrowUpRight, ArrowRight } from 'lucide-react';
import { listLearnTracks } from '@/lib/content';

export const metadata = {
  title: 'Learn | Classical ML From Scratch',
  description:
    'Seven phases that take you from problem brief to deployed model.',
};

export default function LearnPage() {
  const tracks = listLearnTracks();
  const phasesTrack = tracks.find((t) => t.id === 'phases');

  return (
    <>
      {/* Editorial hero */}
      <section className="max-w-4xl mx-auto px-4 md:px-6 pt-16 md:pt-24 pb-10 md:pb-14">
        <div className="text-[10px] font-mono tracking-[0.3em] uppercase text-gray-400 mb-6">
          The Classical ML Pipeline
        </div>
        <h1 className="font-[family-name:var(--font-plex-serif)] text-4xl md:text-6xl lg:text-7xl leading-[1.05] tracking-tight font-normal mb-6">
          Seven phases.
          <br />
          <span className="italic text-emerald-700 dark:text-emerald-300">One playbook.</span>
        </h1>
        <p className="font-[family-name:var(--font-plex-serif)] text-lg md:text-xl leading-snug text-gray-600 dark:text-gray-300 max-w-2xl">
          From reading the brief to deploying the model. Each phase is a chapter — steps,
          decisions, code, and the mistakes that bite hardest.
        </p>
      </section>

      {/* Phase chapters */}
      {phasesTrack && (
        <section className="max-w-4xl mx-auto px-4 md:px-6 pb-16 md:pb-24">
          <ol className="border-t border-gray-200 dark:border-gray-800">
            {phasesTrack.articles.map((a) => {
              const phaseNum = a.slug.match(/phase_(\d+)/)?.[1]?.padStart(2, '0') ?? '··';
              const displayTitle = a.title.replace(/^Phase\s+\d+:\s*/i, '');
              if (!a.exists) {
                return (
                  <li key={a.slug} className="border-b border-gray-200 dark:border-gray-800">
                    <div className="grid grid-cols-[auto,1fr] gap-x-8 md:gap-x-12 py-6 md:py-8 opacity-50">
                      <span className="font-mono text-sm tracking-[0.2em] text-gray-400">
                        {phaseNum}
                      </span>
                      <div>
                        <div className="font-[family-name:var(--font-plex-serif)] text-2xl md:text-3xl mb-1">
                          {displayTitle}
                        </div>
                        {a.summary && (
                          <p className="text-sm text-gray-500 dark:text-gray-400 leading-relaxed max-w-2xl">
                            {a.summary}
                          </p>
                        )}
                        <div className="text-[10px] font-mono tracking-[0.2em] uppercase text-amber-600 dark:text-amber-400 mt-2">
                          Coming soon
                        </div>
                      </div>
                    </div>
                  </li>
                );
              }
              return (
                <li key={a.slug} className="border-b border-gray-200 dark:border-gray-800 group">
                  <Link
                    href={`/learn/${a.slug}`}
                    className="grid grid-cols-[auto,1fr,auto] items-baseline gap-x-6 md:gap-x-12 py-6 md:py-8 hover:bg-emerald-50/30 dark:hover:bg-emerald-950/20 -mx-4 md:-mx-6 px-4 md:px-6 transition-colors"
                  >
                    <span className="font-mono text-sm tracking-[0.2em] text-gray-400 group-hover:text-emerald-700 dark:group-hover:text-emerald-300 transition-colors">
                      {phaseNum}
                    </span>
                    <div className="min-w-0">
                      <div className="font-[family-name:var(--font-plex-serif)] text-2xl md:text-3xl mb-1 group-hover:text-emerald-800 dark:group-hover:text-emerald-200 transition-colors">
                        {displayTitle}
                      </div>
                      {a.summary && (
                        <p className="text-sm text-gray-500 dark:text-gray-400 leading-relaxed max-w-2xl">
                          {a.summary}
                        </p>
                      )}
                    </div>
                    <ArrowUpRight className="w-5 h-5 text-gray-300 dark:text-gray-700 group-hover:text-emerald-600 dark:group-hover:text-emerald-400 group-hover:translate-x-0.5 group-hover:-translate-y-0.5 transition-all flex-shrink-0 self-center" />
                  </Link>
                </li>
              );
            })}
          </ol>

          {/* CTA — start the walkthrough */}
          <div className="mt-12 flex justify-center">
            <Link
              href="/learn/04_phase_1_understand_problem"
              className="group inline-flex items-center gap-3 px-6 py-3 border-2 border-black dark:border-white bg-white dark:bg-gray-900 text-black dark:text-white text-xs uppercase tracking-widest font-semibold hover:bg-emerald-50 dark:hover:bg-emerald-950/40 hover:border-emerald-600 dark:hover:border-emerald-400 hover:text-emerald-800 dark:hover:text-emerald-200 transition-colors"
            >
              Start with Phase 1
              <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
            </Link>
          </div>
        </section>
      )}
    </>
  );
}
