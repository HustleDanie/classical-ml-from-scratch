import Link from 'next/link';
import { ArrowUpRight } from 'lucide-react';
import { listExpertScenarios } from '@/lib/content';

export const metadata = {
  title: 'Expert Scenarios | Classical ML From Scratch',
  description:
    'Comprehensive library of ~155 industry classification + regression case studies — fraud, churn, fairness, calibration, causal uplift, survival, and more.',
};

/**
 * Two semantic accents are used here (emerald + amber). Both come from the
 * documented design palette — classification gets the brand emerald, regression
 * gets the amber accent already reserved for badges. No third colour.
 */
const TYPE_META = {
  classification: {
    label: 'Classification',
    chipClass:
      'border-emerald-500 text-emerald-700 dark:text-emerald-300 hover:bg-emerald-50 dark:hover:bg-emerald-950/30',
    headingClass: 'text-emerald-700 dark:text-emerald-300',
    deepDiveBadgeClass: 'text-emerald-700 dark:text-emerald-300',
    deepDiveBorderClass:
      'border-emerald-300 dark:border-emerald-800 hover:border-emerald-600 dark:hover:border-emerald-400 hover:bg-emerald-50/50 dark:hover:bg-emerald-950/20',
    blurb:
      '79 walkthroughs spanning imbalanced binary, multilabel, hierarchical, ordinal, calibrated, cost-sensitive, real-time, adversarial, regulated, and ethics-bounded scenarios.',
  },
  regression: {
    label: 'Regression',
    chipClass:
      'border-amber-500 text-amber-700 dark:text-amber-300 hover:bg-amber-50 dark:hover:bg-amber-950/30',
    headingClass: 'text-amber-700 dark:text-amber-300',
    deepDiveBadgeClass: 'text-amber-700 dark:text-amber-300',
    deepDiveBorderClass:
      'border-amber-300 dark:border-amber-800 hover:border-amber-600 dark:hover:border-amber-400 hover:bg-amber-50/50 dark:hover:bg-amber-950/20',
    blurb:
      '76 walkthroughs spanning property, financial, healthcare, insurance (Tweedie GLM), manufacturing, energy, time-series, dynamic pricing, CLV, count, quantile, and survival regression.',
  },
} as const;

export default function ScenariosIndexPage() {
  const all = listExpertScenarios();
  const classification = all.filter((s) => s.type === 'classification');
  const regression = all.filter((s) => s.type === 'regression');

  return (
    <>
      <section className="py-12 md:py-16 border-b border-gray-200 dark:border-gray-800">
        <div className="max-w-5xl mx-auto px-4 md:px-6">
          <div className="text-[10px] font-mono tracking-[0.3em] uppercase text-gray-400 mb-2">
            ARCHIVE / {all.length} SCENARIOS
          </div>
          <h1 className="font-orbitron text-3xl md:text-5xl font-bold tracking-tight mb-4">
            EXPERT&nbsp;SCENARIOS
          </h1>
          <p className="text-gray-600 dark:text-gray-300 max-w-3xl text-base md:text-lg leading-relaxed">
            A library of realistic industry problems, organised by archetype. Each scenario
            summarises the brief, recommended approach, primary metric, and watch-outs. Deep
            dives walk through the full pipeline (~500 lines of expert reasoning); compact
            entries focus on the unique delta from a related deep dive.
          </p>

          <div className="mt-6 flex flex-wrap gap-3 text-xs">
            <Link
              href="#classification"
              className={`px-3 py-1.5 border ${TYPE_META.classification.chipClass} transition-colors uppercase tracking-widest`}
            >
              {classification.length} Classification
            </Link>
            <Link
              href="#regression"
              className={`px-3 py-1.5 border ${TYPE_META.regression.chipClass} transition-colors uppercase tracking-widest`}
            >
              {regression.length} Regression
            </Link>
          </div>
        </div>
      </section>

      {(['classification', 'regression'] as const).map((type) => {
        const subset = type === 'classification' ? classification : regression;
        const meta = TYPE_META[type];
        if (subset.length === 0) return null;
        return (
          <section
            key={type}
            id={type}
            className="max-w-7xl mx-auto px-4 md:px-6 py-12 scroll-mt-24"
          >
            <div className="mb-6 border-b border-gray-200 dark:border-gray-800 pb-4">
              <div className="text-[10px] font-mono tracking-[0.3em] uppercase text-gray-400 mb-1">
                CATEGORY
              </div>
              <h2
                className={`font-orbitron text-2xl md:text-3xl font-bold tracking-wider ${meta.headingClass}`}
              >
                {meta.label.toUpperCase()}
                <span className="ml-3 text-xs text-gray-400 font-mono">
                  [{String(subset.length).padStart(2, '0')}]
                </span>
              </h2>
              <p className="text-sm text-gray-500 dark:text-gray-400 mt-2 max-w-3xl leading-relaxed">
                {meta.blurb}
              </p>
              <div className="mt-3 flex flex-wrap gap-4 text-xs">
                <Link
                  href={`/scenarios/${type}/CATALOG`}
                  className="text-gray-500 hover:text-emerald-700 dark:hover:text-emerald-300 transition-colors uppercase tracking-widest"
                >
                  Browse catalog →
                </Link>
                <Link
                  href={`/scenarios/${type}/_methodology`}
                  className="text-gray-500 hover:text-emerald-700 dark:hover:text-emerald-300 transition-colors uppercase tracking-widest"
                >
                  From brief to solution →
                </Link>
              </div>
            </div>

            <ul className="grid grid-cols-1 md:grid-cols-2 gap-3">
              {subset.map((s) => {
                const borderClass = s.isDeepDive
                  ? meta.deepDiveBorderClass
                  : 'border-gray-200 dark:border-gray-800 hover:border-emerald-600 dark:hover:border-emerald-400 hover:bg-emerald-50/50 dark:hover:bg-emerald-950/20';
                return (
                  <li key={`${s.type}/${s.slug}`}>
                    <Link
                      href={`/scenarios/${s.type}/${s.slug}`}
                      className={`group flex h-full flex-col border px-4 py-3 transition-all duration-200 hover:-translate-y-0.5 hover:shadow-md dark:hover:shadow-emerald-950/40 ${borderClass}`}
                    >
                      <div className="flex items-start gap-3">
                        <span className="font-mono text-[10px] tracking-widest text-gray-400 mt-0.5">
                          {s.num}
                        </span>
                        <div className="flex-1 min-w-0">
                          <div className="flex items-start justify-between gap-2 mb-1">
                            <div className="text-sm font-semibold leading-tight">
                              {s.title}
                            </div>
                            <ArrowUpRight className="w-3.5 h-3.5 text-gray-300 dark:text-gray-700 group-hover:text-emerald-600 dark:group-hover:text-emerald-400 group-hover:translate-x-0.5 group-hover:-translate-y-0.5 transition-all flex-shrink-0 mt-0.5" />
                          </div>
                          {s.summary && (
                            <p className="text-xs text-gray-500 dark:text-gray-400 leading-relaxed line-clamp-3">
                              {s.summary}
                            </p>
                          )}
                          {s.isDeepDive && (
                            <span
                              className={`inline-block mt-2 text-[9px] tracking-widest uppercase ${meta.deepDiveBadgeClass}`}
                            >
                              ▸ Deep dive
                            </span>
                          )}
                        </div>
                      </div>
                    </Link>
                  </li>
                );
              })}
            </ul>
          </section>
        );
      })}
    </>
  );
}
