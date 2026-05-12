import Link from 'next/link';
import { ArrowUpRight, Layers, FileSearch } from 'lucide-react';

export const metadata = {
  title: 'Learn | Classical ML From Scratch',
  description:
    'Seven phases that take you from problem brief to deployed model — plus a worked example of reading a real brief.',
};

export default function LearnPage() {
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
          From reading the brief to deploying the model. Two ways in: walk the phases,
          or watch one expert dissect a real brief into phase decisions.
        </p>
      </section>

      {/* Two cards */}
      <section className="max-w-4xl mx-auto px-4 md:px-6 pb-16 md:pb-24">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-px bg-gray-200 dark:bg-gray-800 border border-gray-200 dark:border-gray-800">
          {/* Card 1 — Pipeline Phases */}
          <Link
            href="/learn/04_phase_1_understand_problem"
            className="group block bg-white dark:bg-gray-900 hover:bg-emerald-50/60 dark:hover:bg-emerald-950/30 transition-colors p-8 md:p-10"
          >
            <div className="flex items-start justify-between mb-5">
              <div className="text-[10px] font-mono tracking-[0.3em] uppercase text-gray-400 group-hover:text-emerald-700 dark:group-hover:text-emerald-300 transition-colors">
                01 / Walkthrough
              </div>
              <Layers className="w-5 h-5 text-gray-400 group-hover:text-emerald-600 dark:group-hover:text-emerald-400 transition-colors" />
            </div>
            <div className="font-[family-name:var(--font-plex-serif)] text-3xl md:text-4xl font-normal leading-tight mb-3">
              Pipeline Phases
            </div>
            <p className="text-sm md:text-base text-gray-600 dark:text-gray-300 leading-relaxed mb-6 max-w-sm">
              Seven chapters, in order. Each phase has its steps, decision tables, the
              code that is the answer, and the mistakes that bite hardest.
            </p>
            <div className="flex items-center justify-between text-xs uppercase tracking-widest font-semibold">
              <span className="group-hover:text-emerald-700 dark:group-hover:text-emerald-300 transition-colors">
                Start with Phase 1
              </span>
              <ArrowUpRight className="w-5 h-5 text-gray-300 dark:text-gray-700 group-hover:text-emerald-600 dark:group-hover:text-emerald-400 group-hover:translate-x-0.5 group-hover:-translate-y-0.5 transition-all" />
            </div>
          </Link>

          {/* Card 2 — Reading a Brief */}
          <Link
            href="/learn/brief-reading"
            className="group block bg-white dark:bg-gray-900 hover:bg-emerald-50/60 dark:hover:bg-emerald-950/30 transition-colors p-8 md:p-10"
          >
            <div className="flex items-start justify-between mb-5">
              <div className="text-[10px] font-mono tracking-[0.3em] uppercase text-gray-400 group-hover:text-emerald-700 dark:group-hover:text-emerald-300 transition-colors">
                02 / Worked example
              </div>
              <FileSearch className="w-5 h-5 text-gray-400 group-hover:text-emerald-600 dark:group-hover:text-emerald-400 transition-colors" />
            </div>
            <div className="font-[family-name:var(--font-plex-serif)] text-3xl md:text-4xl font-normal leading-tight mb-3">
              Reading a Brief
            </div>
            <p className="text-sm md:text-base text-gray-600 dark:text-gray-300 leading-relaxed mb-6 max-w-sm">
              The sample brief from the practice page, dissected. For each phase: the
              exact phrases an expert pulls from the text, and the decision each one
              forces.
            </p>
            <div className="flex items-center justify-between text-xs uppercase tracking-widest font-semibold">
              <span className="group-hover:text-emerald-700 dark:group-hover:text-emerald-300 transition-colors">
                See the dissection
              </span>
              <ArrowUpRight className="w-5 h-5 text-gray-300 dark:text-gray-700 group-hover:text-emerald-600 dark:group-hover:text-emerald-400 group-hover:translate-x-0.5 group-hover:-translate-y-0.5 transition-all" />
            </div>
          </Link>
        </div>

        {/* Compact phase-jump list */}
        <div className="mt-10 pt-8 border-t border-gray-200 dark:border-gray-800">
          <div className="text-[10px] font-mono tracking-[0.3em] uppercase text-gray-400 mb-4">
            Or jump to a specific phase
          </div>
          <ol className="grid grid-cols-1 md:grid-cols-2 gap-x-6 gap-y-1">
            {[
              { num: '01', slug: '04_phase_1_understand_problem', title: 'Understand the Problem' },
              { num: '02', slug: '05_phase_2_data_exploration_cleaning', title: 'Data Exploration & Cleaning' },
              { num: '03', slug: '06_phase_3_feature_selection_preprocessing', title: 'Feature Selection & Preprocessing' },
              { num: '04', slug: '07_phase_4_model_selection_training', title: 'Model Selection & Training' },
              { num: '05', slug: '08_phase_5_optimization', title: 'Optimization' },
              { num: '06', slug: '09_phase_6_evaluation_validation', title: 'Evaluation & Validation' },
              { num: '07', slug: '10_phase_7_deployment', title: 'Deployment' },
            ].map((p) => (
              <li key={p.slug}>
                <Link
                  href={`/learn/${p.slug}`}
                  className="group flex items-baseline gap-4 py-2 text-sm hover:text-emerald-700 dark:hover:text-emerald-300 transition-colors"
                >
                  <span className="font-mono text-[10px] tracking-[0.2em] text-gray-400 w-7 flex-shrink-0">
                    {p.num}
                  </span>
                  <span className="text-gray-700 dark:text-gray-200 group-hover:text-emerald-800 dark:group-hover:text-emerald-200 transition-colors">
                    {p.title}
                  </span>
                </Link>
              </li>
            ))}
          </ol>
        </div>
      </section>
    </>
  );
}
