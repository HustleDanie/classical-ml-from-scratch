import Link from 'next/link';

export const PHASES: { num: string; slug: string; title: string }[] = [
  { num: '01', slug: '04_phase_1_understand_problem', title: 'Understand the Problem' },
  { num: '02', slug: '05_phase_2_data_exploration_cleaning', title: 'Data Exploration & Cleaning' },
  { num: '03', slug: '06_phase_3_feature_selection_preprocessing', title: 'Feature Selection & Preprocessing' },
  { num: '04', slug: '07_phase_4_model_selection_training', title: 'Model Selection & Training' },
  { num: '05', slug: '08_phase_5_optimization', title: 'Optimization' },
  { num: '06', slug: '09_phase_6_evaluation_validation', title: 'Evaluation & Validation' },
  { num: '07', slug: '10_phase_7_deployment', title: 'Deployment' },
];

interface PhaseNavProps {
  currentSlug: string;
}

/**
 * Sticky left-rail nav for the seven phase deep-dive pages.
 * Hidden below lg breakpoint (use PhaseProgress dot strip there instead).
 */
export function PhaseNav({ currentSlug }: PhaseNavProps) {
  return (
    <aside
      aria-label="Pipeline phases"
      className="hidden lg:block sticky top-24 self-start w-56 flex-shrink-0"
    >
      <div className="text-[10px] font-mono tracking-[0.3em] uppercase text-gray-400 mb-4">
        Pipeline phases
      </div>
      <ol className="space-y-0.5">
        {PHASES.map((p) => {
          const isCurrent = p.slug === currentSlug;
          return (
            <li key={p.slug}>
              <Link
                href={`/learn/${p.slug}`}
                aria-current={isCurrent ? 'page' : undefined}
                className={`group flex items-baseline gap-3 py-1.5 text-sm transition-colors ${
                  isCurrent
                    ? 'text-emerald-700 dark:text-emerald-300 font-semibold'
                    : 'text-gray-500 dark:text-gray-400 hover:text-black dark:hover:text-white'
                }`}
              >
                <span
                  className={`font-mono text-[10px] tracking-[0.2em] w-7 flex-shrink-0 ${
                    isCurrent
                      ? 'text-emerald-700 dark:text-emerald-300'
                      : 'text-gray-400 dark:text-gray-500'
                  }`}
                >
                  {p.num}
                </span>
                <span className="leading-tight">{p.title}</span>
              </Link>
            </li>
          );
        })}
      </ol>
    </aside>
  );
}
