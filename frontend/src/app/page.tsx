import Link from 'next/link';
import { ArrowRight } from 'lucide-react';

export default function Home() {
  return (
    <>
      {/* Editorial hero */}
      <section className="max-w-4xl mx-auto px-4 md:px-6 pt-20 md:pt-32 pb-12 md:pb-16">
        <div className="text-[10px] font-mono tracking-[0.3em] uppercase text-gray-400 mb-6">
          · numpy · sklearn-verified ·
        </div>
        <h1 className="font-[family-name:var(--font-plex-serif)] text-5xl md:text-7xl lg:text-[5.5rem] leading-[1.02] tracking-tight font-normal mb-8">
          classical&nbsp;ml,
          <br />
          <span className="italic text-emerald-700 dark:text-emerald-300">from scratch.</span>
        </h1>
        <p className="font-[family-name:var(--font-plex-serif)] text-xl md:text-2xl leading-snug text-gray-600 dark:text-gray-300 max-w-2xl">
          Seven phases. From problem brief to deployed model. Everything else is noise.
        </p>
      </section>

      {/* Two editorial entry points */}
      <section className="max-w-4xl mx-auto px-4 md:px-6 pb-24">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-px bg-gray-200 dark:bg-gray-800 border border-gray-200 dark:border-gray-800">
          {/* LEARN */}
          <Link
            href="/learn"
            className="group block bg-white dark:bg-gray-900 hover:bg-emerald-50/60 dark:hover:bg-emerald-950/30 transition-colors p-8 md:p-10"
          >
            <div className="text-[10px] font-mono tracking-[0.3em] uppercase text-gray-400 group-hover:text-emerald-700 dark:group-hover:text-emerald-300 transition-colors mb-3">
              01 / Read
            </div>
            <div className="font-[family-name:var(--font-plex-serif)] text-3xl md:text-4xl font-normal leading-tight mb-3">
              Learn
            </div>
            <p className="text-sm md:text-base text-gray-600 dark:text-gray-300 leading-relaxed mb-6 max-w-sm">
              Seven phase deep-dives. Step-by-step decisions, common mistakes,
              the code that is the answer.
            </p>
            <div className="inline-flex items-center gap-2 text-xs uppercase tracking-widest font-semibold group-hover:text-emerald-700 dark:group-hover:text-emerald-300 transition-colors">
              Start with Phase 1
              <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
            </div>
          </Link>

          {/* PRACTICE */}
          <Link
            href="/practice"
            className="group block bg-white dark:bg-gray-900 hover:bg-emerald-50/60 dark:hover:bg-emerald-950/30 transition-colors p-8 md:p-10"
          >
            <div className="text-[10px] font-mono tracking-[0.3em] uppercase text-gray-400 group-hover:text-emerald-700 dark:group-hover:text-emerald-300 transition-colors mb-3">
              02 / Apply
            </div>
            <div className="font-[family-name:var(--font-plex-serif)] text-3xl md:text-4xl font-normal leading-tight mb-3">
              Practice
            </div>
            <p className="text-sm md:text-base text-gray-600 dark:text-gray-300 leading-relaxed mb-6 max-w-sm">
              A real brief. Work through it on paper. Reveal a 17-step
              expert solution to compare.
            </p>
            <div className="inline-flex items-center gap-2 text-xs uppercase tracking-widest font-semibold group-hover:text-emerald-700 dark:group-hover:text-emerald-300 transition-colors">
              Load a brief
              <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
            </div>
          </Link>
        </div>
      </section>
    </>
  );
}
