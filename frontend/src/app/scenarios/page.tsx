import Link from 'next/link';
import { listExpertScenarios } from '@/lib/content';

export const metadata = {
  title: 'Expert Scenarios | Classical ML From Scratch',
  description:
    'Comprehensive library of ~155 industry classification + regression case studies — fraud, churn, fairness, calibration, causal uplift, survival, and more.',
};

const TYPE_META: Record<
  'classification' | 'regression',
  { label: string; accent: string; blurb: string }
> = {
  classification: {
    label: 'Classification',
    accent: 'border-emerald-500 text-emerald-500',
    blurb:
      '79 walkthroughs spanning imbalanced binary, multilabel, hierarchical, ordinal, calibrated, cost-sensitive, real-time, adversarial, regulated, and ethics-bounded scenarios.',
  },
  regression: {
    label: 'Regression',
    accent: 'border-blue-500 text-blue-500',
    blurb:
      '76 walkthroughs spanning property, financial, healthcare, insurance (Tweedie GLM), manufacturing, energy, time-series, dynamic pricing, CLV, count, quantile, and survival regression.',
  },
};

export default function ScenariosIndexPage() {
  const all = listExpertScenarios();
  const classification = all.filter((s) => s.type === 'classification');
  const regression = all.filter((s) => s.type === 'regression');

  return (
    <>
      <section className="relative py-12 md:py-16 overflow-hidden border-b border-gray-200 dark:border-gray-800">
        <div className="relative max-w-5xl mx-auto px-4 md:px-6">
          <div className="text-[10px] font-mono tracking-[0.3em] uppercase text-gray-400 mb-2">
            ARCHIVE / {all.length} SCENARIOS
          </div>
          <h1 className="font-orbitron text-3xl md:text-5xl font-bold tracking-tight mb-4">
            EXPERT&nbsp;SCENARIOS
          </h1>
          <p className="text-gray-600 dark:text-gray-300 max-w-3xl text-base md:text-lg leading-relaxed">
            A library of realistic industry problems, organised by archetype. Each scenario summarises
            the brief, recommended approach, primary metric, and watch-outs. Deep dives walk through
            the full pipeline (~500 lines of expert reasoning); compact entries focus on the unique
            delta from a related deep dive.
          </p>

          <div className="mt-6 flex flex-wrap gap-3 text-xs">
            <Link
              href="#classification"
              className="px-3 py-1.5 border border-emerald-500 text-emerald-500 hover:bg-emerald-500/10 transition-colors uppercase tracking-widest"
            >
              {classification.length} Classification
            </Link>
            <Link
              href="#regression"
              className="px-3 py-1.5 border border-blue-500 text-blue-500 hover:bg-blue-500/10 transition-colors uppercase tracking-widest"
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
                className={`font-orbitron text-2xl md:text-3xl font-bold tracking-wider ${meta.accent.split(' ')[1]}`}
              >
                {meta.label.toUpperCase()}
                <span className="ml-3 text-xs text-gray-400 font-mono">
                  [{String(subset.length).padStart(2, '0')}]
                </span>
              </h2>
              <p className="text-sm text-gray-500 dark:text-gray-400 mt-2 max-w-3xl">
                {meta.blurb}
              </p>
              <div className="mt-3 text-xs">
                <Link
                  href={`/scenarios/${type}/CATALOG`}
                  className="mr-4 text-gray-500 hover:text-black dark:hover:text-white transition-colors uppercase tracking-widest"
                >
                  Browse catalog →
                </Link>
                <Link
                  href={`/scenarios/${type}/_methodology`}
                  className="text-gray-500 hover:text-black dark:hover:text-white transition-colors uppercase tracking-widest"
                >
                  From brief to solution →
                </Link>
              </div>
            </div>

            <ul className="grid grid-cols-1 md:grid-cols-2 gap-3">
              {subset.map((s) => (
                <li key={`${s.type}/${s.slug}`}>
                  <Link
                    href={`/scenarios/${s.type}/${s.slug}`}
                    className={`relative block border border-gray-200 dark:border-gray-800 px-4 py-3 hover:border-current transition-colors h-full ${
                      s.isDeepDive ? meta.accent : ''
                    }`}
                  >
                    <div className="flex items-start gap-3">
                      <span className="font-mono text-[10px] tracking-widest text-gray-400 mt-0.5">
                        {s.num}
                      </span>
                      <div className="flex-1">
                        <div className="text-sm font-semibold leading-tight mb-1">
                          {s.title}
                        </div>
                        {s.summary && (
                          <p className="text-xs text-gray-500 dark:text-gray-400 leading-relaxed line-clamp-3">
                            {s.summary}
                          </p>
                        )}
                        {s.isDeepDive && (
                          <span
                            className={`inline-block mt-2 text-[9px] tracking-widest uppercase ${meta.accent.split(' ')[1]}`}
                          >
                            ▸ Deep dive
                          </span>
                        )}
                      </div>
                    </div>
                  </Link>
                </li>
              ))}
            </ul>
          </section>
        );
      })}
    </>
  );
}
