import Link from 'next/link';
import { PHASES } from './PhaseNav';

interface PhaseProgressProps {
  currentSlug: string;
}

/**
 * 7-dot progress strip — shows where the reader is in the pipeline.
 * Visible at all viewport sizes; complements the sticky PhaseNav (lg+).
 */
export function PhaseProgress({ currentSlug }: PhaseProgressProps) {
  const currentIndex = PHASES.findIndex((p) => p.slug === currentSlug);
  if (currentIndex < 0) return null;

  return (
    <div className="flex items-center gap-3">
      <div className="flex items-center gap-1.5">
        {PHASES.map((p, i) => {
          const isCurrent = i === currentIndex;
          const isPast = i < currentIndex;
          return (
            <Link
              key={p.slug}
              href={`/learn/${p.slug}`}
              aria-label={`Phase ${p.num} — ${p.title}`}
              className="block group"
            >
              <span
                className={`block w-2 h-2 rounded-full transition-all ${
                  isCurrent
                    ? 'bg-emerald-600 dark:bg-emerald-400 ring-2 ring-emerald-600/30 dark:ring-emerald-400/30 ring-offset-1 ring-offset-white dark:ring-offset-gray-950'
                    : isPast
                    ? 'bg-gray-400 dark:bg-gray-500 group-hover:bg-emerald-500'
                    : 'bg-gray-200 dark:bg-gray-700 group-hover:bg-gray-400 dark:group-hover:bg-gray-500'
                }`}
              />
            </Link>
          );
        })}
      </div>
      <span className="text-[10px] font-mono tracking-[0.2em] uppercase text-gray-400">
        Phase {PHASES[currentIndex].num} of 07
      </span>
    </div>
  );
}
