import { PIPELINES } from '@/lib/pipelines';
import { PipelineCard } from '@/components/PipelineCard';

export const metadata = {
  title: 'Pipelines | Classical ML From Scratch',
  description: 'Three end-to-end ML pipelines on real, messy datasets — Titanic, Ames Housing, 20 Newsgroups.',
};

export default function PipelinesIndexPage() {
  return (
    <>
      <section className="relative py-12 md:py-16 overflow-hidden border-b border-gray-200 dark:border-gray-800">
        <div className="relative max-w-5xl mx-auto px-4 md:px-6">
          <div className="text-[10px] font-mono tracking-[0.3em] uppercase text-gray-400 mb-2">
            END-TO-END / PRODUCTION-LIKE
          </div>
          <h1 className="font-orbitron text-3xl md:text-5xl font-bold tracking-tight mb-4">
            REAL-WORLD PIPELINES
          </h1>
          <p className="text-gray-600 dark:text-gray-300 max-w-2xl">
            Each pipeline tackles a different kind of messy real-world data — from missing values and
            class imbalance, to skewed targets and high-dimensional text — and walks through the
            full ML lifecycle: cleaning, feature engineering, modelling, tuning, ensembling, and explainability.
          </p>
        </div>
      </section>

      <section className="max-w-7xl mx-auto px-4 md:px-6 py-12">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {PIPELINES.map((pipeline, index) => (
            <PipelineCard key={pipeline.slug} pipeline={pipeline} index={index} />
          ))}
        </div>
      </section>
    </>
  );
}
