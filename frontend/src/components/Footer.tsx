import Link from 'next/link';

export function Footer() {
  return (
    <footer className="border-t border-gray-200 dark:border-gray-800 mt-16">
      <div className="max-w-7xl mx-auto px-4 md:px-6 py-6 flex flex-col md:flex-row md:items-center md:justify-between gap-3">
        <div className="flex items-center gap-3 text-[10px] font-mono tracking-[0.3em] uppercase text-gray-400">
          <span>· numpy · sklearn-verified ·</span>
        </div>
        <div className="flex flex-wrap items-center gap-x-5 gap-y-1 text-xs">
          <Link
            href="/learn"
            className="text-gray-500 hover:text-emerald-700 dark:hover:text-emerald-300 transition-colors uppercase tracking-widest"
          >
            Learn
          </Link>
          <Link
            href="/practice"
            className="text-gray-500 hover:text-emerald-700 dark:hover:text-emerald-300 transition-colors uppercase tracking-widest"
          >
            Practice
          </Link>
          <Link
            href="/scenarios"
            className="text-gray-500 hover:text-emerald-700 dark:hover:text-emerald-300 transition-colors uppercase tracking-widest"
          >
            Scenarios
          </Link>
          <Link
            href="/pipelines"
            className="text-gray-500 hover:text-emerald-700 dark:hover:text-emerald-300 transition-colors uppercase tracking-widest"
          >
            Pipelines
          </Link>
          <a
            href="https://github.com/HustleDanie/classical-ml-from-scratch"
            target="_blank"
            rel="noopener noreferrer"
            className="text-gray-500 hover:text-emerald-700 dark:hover:text-emerald-300 transition-colors uppercase tracking-widest"
          >
            GitHub
          </a>
        </div>
      </div>
    </footer>
  );
}
