import Link from 'next/link';
import { BriefReadingClient } from '@/components/BriefReadingClient';

export const metadata = {
  title: 'Reading a Brief | Learn — Classical ML From Scratch',
  description:
    'Generate a brief, its matching dataset, and the exact phrases an expert pulls out for each pipeline phase — all in one call.',
};

export default function BriefReadingPage() {
  return (
    <>
      {/* Breadcrumb */}
      <section className="border-b border-gray-200 dark:border-gray-800">
        <div className="max-w-5xl mx-auto px-4 md:px-6 py-4 flex items-center gap-2 text-[10px] font-mono tracking-[0.2em] uppercase text-gray-400">
          <Link
            href="/learn"
            className="hover:text-emerald-700 dark:hover:text-emerald-300 transition-colors"
          >
            Learn
          </Link>
          <span className="text-gray-300 dark:text-gray-700">/</span>
          <span className="text-black dark:text-white">Reading a Brief</span>
        </div>
      </section>

      {/* Editorial hero */}
      <section className="max-w-4xl mx-auto px-4 md:px-6 pt-12 md:pt-16 pb-8 md:pb-10">
        <div className="text-[10px] font-mono tracking-[0.3em] uppercase text-gray-400 mb-4">
          One brief · Seven phases
        </div>
        <h1 className="font-[family-name:var(--font-plex-serif)] text-4xl md:text-6xl leading-[1.05] tracking-tight font-normal mb-4">
          Reading a Brief.
        </h1>
        <p className="font-[family-name:var(--font-plex-serif)] text-lg md:text-xl leading-snug text-gray-600 dark:text-gray-300 max-w-3xl">
          Generate a fresh brief, its matching dataset, and the seven-phase dissection in
          one shot. For every phase the model pulls back the exact phrases from the brief
          and the decision each one forces.
        </p>
      </section>

      {/* Interactive — generate, view, regenerate */}
      <BriefReadingClient />
    </>
  );
}
