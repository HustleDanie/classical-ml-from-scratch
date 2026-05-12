import Link from 'next/link';
import { BookOpen, Sparkles, ArrowRight } from 'lucide-react';

export default function Home() {
  return (
    <>
      {/* Hero — plain background, no grid overlay */}
      <section className="relative py-16 md:py-24">
        <div className="relative max-w-4xl mx-auto px-4 md:px-6 text-center">
          <div className="inline-block mb-6">
            <span className="px-3 py-1 text-[10px] font-mono tracking-[0.3em] uppercase border border-black dark:border-white">
              · numpy · sklearn-verified ·
            </span>
          </div>
          <h2 className="font-orbitron text-4xl md:text-6xl lg:text-7xl font-bold mb-4 tracking-tight">
            CLASSICAL&nbsp;ML
          </h2>
          <div className="w-16 h-0.5 bg-black dark:bg-white mx-auto mb-6" />
          <p className="text-gray-600 dark:text-gray-300 text-base md:text-lg max-w-2xl mx-auto leading-relaxed">
            Two ways in. Pick one.
          </p>
        </div>
      </section>

      {/* Two cards — matching colours, subtle blue accent on hover */}
      <section className="max-w-6xl mx-auto px-4 md:px-6 pb-24">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 md:gap-8">
          {/* LEARN card */}
          <Link
            href="/learn"
            className="group relative block border-2 border-black dark:border-white bg-white dark:bg-gray-900 text-black dark:text-white p-8 md:p-10 hover:bg-emerald-50 dark:hover:bg-emerald-950/40 hover:border-emerald-600 dark:hover:border-emerald-400 hover:-translate-y-1 hover:shadow-xl dark:hover:shadow-emerald-950/50 transition-all duration-200"
          >
            <div className="absolute -top-1 -left-1 w-3 h-3 border-t-2 border-l-2 border-black dark:border-white group-hover:border-emerald-600 dark:group-hover:border-emerald-400" />
            <div className="absolute -bottom-1 -right-1 w-3 h-3 border-b-2 border-r-2 border-black dark:border-white group-hover:border-emerald-600 dark:group-hover:border-emerald-400" />

            <div className="flex items-center justify-between mb-6">
              <div className="text-[10px] font-mono tracking-[0.3em] uppercase text-gray-400 group-hover:text-emerald-600 dark:group-hover:text-emerald-400 transition-colors">
                01 / Path
              </div>
              <BookOpen className="w-6 h-6 group-hover:text-emerald-600 dark:group-hover:text-emerald-400 transition-colors" />
            </div>

            <h3 className="font-orbitron text-3xl md:text-4xl font-bold tracking-tight mb-4">
              LEARN
            </h3>

            <p className="text-sm md:text-base leading-relaxed mb-8 opacity-80">
              10 chapters across foundations and the seven pipeline phases — a complete
              walk-through for tackling any classical ML problem from brief to deployment.
            </p>

            <div className="flex items-center gap-2 text-xs uppercase tracking-widest font-semibold group-hover:text-emerald-600 dark:group-hover:text-emerald-400 transition-colors">
              Start learning
              <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
            </div>
          </Link>

          {/* PRACTICE card */}
          <Link
            href="/practice"
            className="group relative block border-2 border-black dark:border-white bg-white dark:bg-gray-900 text-black dark:text-white p-8 md:p-10 hover:bg-emerald-50 dark:hover:bg-emerald-950/40 hover:border-emerald-600 dark:hover:border-emerald-400 hover:-translate-y-1 hover:shadow-xl dark:hover:shadow-emerald-950/50 transition-all duration-200"
          >
            <div className="absolute -top-1 -left-1 w-3 h-3 border-t-2 border-l-2 border-black dark:border-white group-hover:border-emerald-600 dark:group-hover:border-emerald-400" />
            <div className="absolute -bottom-1 -right-1 w-3 h-3 border-b-2 border-r-2 border-black dark:border-white group-hover:border-emerald-600 dark:group-hover:border-emerald-400" />

            <div className="flex items-center justify-between mb-6">
              <div className="text-[10px] font-mono tracking-[0.3em] uppercase text-gray-400 group-hover:text-emerald-600 dark:group-hover:text-emerald-400 transition-colors">
                02 / Active
              </div>
              <Sparkles className="w-6 h-6 group-hover:text-emerald-600 dark:group-hover:text-emerald-400 transition-colors" />
            </div>

            <h3 className="font-orbitron text-3xl md:text-4xl font-bold tracking-tight mb-4">
              PRACTICE
            </h3>

            <p className="text-sm md:text-base leading-relaxed mb-8 opacity-80">
              Work through a real ML problem brief end-to-end. Write your own approach,
              then reveal a 17-step expert walkthrough and compare your thinking.
            </p>

            <div className="flex items-center gap-2 text-xs uppercase tracking-widest font-semibold group-hover:text-emerald-600 dark:group-hover:text-emerald-400 transition-colors">
              Start a brief
              <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
            </div>
          </Link>
        </div>
      </section>
    </>
  );
}
