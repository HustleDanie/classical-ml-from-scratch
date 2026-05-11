import { ExternalLink } from 'lucide-react';
import type { Paper } from '@/lib/learning/types';

interface PaperCardProps {
  paper: Paper;
}

const TAG_ACCENT: Record<NonNullable<Paper['tag']>, string> = {
  Seminal: 'border-amber-500 text-amber-500',
  Survey: 'border-cyan-500 text-cyan-500',
  Modern: 'border-emerald-500 text-emerald-500',
  Tutorial: 'border-purple-500 text-purple-500',
};

export function PaperCard({ paper }: PaperCardProps) {
  const meta = [paper.authors, String(paper.year), paper.venue].filter(Boolean).join(' · ');
  const inner = (
    <>
      <div className="flex items-start justify-between gap-3 mb-2 flex-wrap">
        <div className="flex items-center gap-2 flex-wrap">
          {paper.tag && (
            <span
              className={`border ${TAG_ACCENT[paper.tag]} px-2 py-0.5 font-mono text-[10px] tracking-widest`}
            >
              {paper.tag.toUpperCase()}
            </span>
          )}
          <span className="text-[10px] font-mono tracking-widest uppercase text-gray-400">
            {meta}
          </span>
        </div>
        {paper.url && (
          <span className="text-[10px] font-mono tracking-widest uppercase text-gray-400 inline-flex items-center gap-1">
            Open <ExternalLink className="w-3 h-3" />
          </span>
        )}
      </div>
      <h3 className="font-orbitron text-base md:text-lg font-bold tracking-wide leading-snug mb-2">
        {paper.title}
      </h3>
      <p className="text-sm text-gray-600 dark:text-gray-300 leading-relaxed">
        {paper.blurb}
      </p>
    </>
  );

  if (paper.url) {
    return (
      <a
        href={paper.url}
        target="_blank"
        rel="noopener noreferrer"
        className="block border border-gray-200 dark:border-gray-800 p-5 md:p-6 hover:border-black dark:hover:border-white transition-colors no-underline"
      >
        {inner}
      </a>
    );
  }

  return (
    <div className="border border-gray-200 dark:border-gray-800 p-5 md:p-6">{inner}</div>
  );
}
