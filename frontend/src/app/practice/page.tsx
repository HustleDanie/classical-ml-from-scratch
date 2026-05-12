import { PracticeWorkspace } from '@/components/PracticeWorkspace';

export const metadata = {
  title: 'Practice | Classical ML From Scratch',
  description:
    'Walk through a sample ML problem brief. Write your approach, then reveal a 17-step expert walkthrough.',
};

export default function PracticePage() {
  return (
    <>
      <section className="relative py-12 md:py-16 overflow-hidden border-b border-gray-200 dark:border-gray-800">
        <div className="relative max-w-4xl mx-auto px-4 md:px-6">
          <div className="text-[10px] font-mono tracking-[0.3em] uppercase text-gray-400 mb-2">
            ACTIVE LEARNING
          </div>
          <h1 className="font-orbitron text-3xl md:text-5xl font-bold tracking-tight mb-4">
            PRACTICE
          </h1>
          <p className="text-gray-600 dark:text-gray-300 max-w-3xl text-base md:text-lg leading-relaxed">
            The expert scenarios show what good answers look like. This page makes
            you write your own first. Load the sample brief, sketch your approach,
            then reveal a 17-step expert walkthrough so you can compare your thinking
            against an expert&apos;s.
          </p>

          <div className="mt-6 grid grid-cols-1 md:grid-cols-3 gap-3 text-xs">
            <div className="border border-gray-200 dark:border-gray-800 px-3 py-2">
              <div className="font-mono text-[10px] tracking-widest uppercase text-gray-400 mb-1">
                01
              </div>
              <div className="text-gray-700 dark:text-gray-200">Load the sample brief.</div>
            </div>
            <div className="border border-gray-200 dark:border-gray-800 px-3 py-2">
              <div className="font-mono text-[10px] tracking-widest uppercase text-gray-400 mb-1">
                02
              </div>
              <div className="text-gray-700 dark:text-gray-200">Write your approach.</div>
            </div>
            <div className="border border-gray-200 dark:border-gray-800 px-3 py-2">
              <div className="font-mono text-[10px] tracking-widest uppercase text-gray-400 mb-1">
                03
              </div>
              <div className="text-gray-700 dark:text-gray-200">
                Compare against the expert answer.
              </div>
            </div>
          </div>
        </div>
      </section>

      <section className="max-w-4xl mx-auto px-4 md:px-6 py-10">
        <PracticeWorkspace />
      </section>
    </>
  );
}
