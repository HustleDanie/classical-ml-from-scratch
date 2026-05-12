import Link from 'next/link';
import { Cpu, ArrowUpRight } from 'lucide-react';

export function Footer() {
  return (
    <footer className="border-t border-gray-200 dark:border-gray-800 mt-16">
      <div className="max-w-7xl mx-auto px-4 md:px-6 py-10">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8 md:gap-12">
          {/* Brand */}
          <div>
            <div className="flex items-center gap-3 mb-3">
              <div className="w-8 h-8 border border-black dark:border-white flex items-center justify-center">
                <Cpu className="w-4 h-4" />
              </div>
              <div>
                <div className="font-orbitron text-sm font-bold tracking-wider">
                  CLASSICAL ML
                </div>
                <div className="text-[10px] text-gray-500 dark:text-gray-400 tracking-widest uppercase">
                  From Scratch
                </div>
              </div>
            </div>
            <p className="text-xs text-gray-500 dark:text-gray-400 leading-relaxed max-w-xs">
              A beginner-grade walkthrough of the classical ML pipeline — seven phases,
              155 expert scenarios, and three end-to-end production pipelines on real
              datasets.
            </p>
          </div>

          {/* Explore */}
          <div>
            <div className="text-[10px] font-mono tracking-[0.3em] uppercase text-gray-400 mb-3">
              Explore
            </div>
            <ul className="space-y-2 text-sm">
              <li>
                <Link
                  href="/learn"
                  className="text-gray-600 dark:text-gray-300 hover:text-emerald-700 dark:hover:text-emerald-300 transition-colors"
                >
                  The pipeline guide
                </Link>
              </li>
              <li>
                <Link
                  href="/practice"
                  className="text-gray-600 dark:text-gray-300 hover:text-emerald-700 dark:hover:text-emerald-300 transition-colors"
                >
                  Practice with a sample brief
                </Link>
              </li>
              <li>
                <Link
                  href="/scenarios"
                  className="text-gray-600 dark:text-gray-300 hover:text-emerald-700 dark:hover:text-emerald-300 transition-colors"
                >
                  155 expert scenarios
                </Link>
              </li>
              <li>
                <Link
                  href="/pipelines"
                  className="text-gray-600 dark:text-gray-300 hover:text-emerald-700 dark:hover:text-emerald-300 transition-colors"
                >
                  3 production pipelines
                </Link>
              </li>
            </ul>
          </div>

          {/* Source */}
          <div>
            <div className="text-[10px] font-mono tracking-[0.3em] uppercase text-gray-400 mb-3">
              Source
            </div>
            <ul className="space-y-2 text-sm">
              <li>
                <a
                  href="https://github.com/HustleDanie/classical-ml-from-scratch"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center gap-1.5 text-gray-600 dark:text-gray-300 hover:text-emerald-700 dark:hover:text-emerald-300 transition-colors"
                >
                  GitHub repository
                  <ArrowUpRight className="w-3.5 h-3.5" />
                </a>
              </li>
              <li>
                <a
                  href="https://github.com/HustleDanie/classical-ml-from-scratch/blob/master/README.md"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center gap-1.5 text-gray-600 dark:text-gray-300 hover:text-emerald-700 dark:hover:text-emerald-300 transition-colors"
                >
                  README
                  <ArrowUpRight className="w-3.5 h-3.5" />
                </a>
              </li>
            </ul>
          </div>
        </div>

        <div className="mt-10 pt-6 border-t border-gray-200 dark:border-gray-800 flex flex-col md:flex-row justify-between items-start md:items-center gap-3">
          <p className="text-[10px] font-mono tracking-widest uppercase text-gray-400">
            · numpy · sklearn-verified ·
          </p>
          <p className="text-[10px] text-gray-400 dark:text-gray-500">
            Built with Next.js · Tailwind · IBM Plex
          </p>
        </div>
      </div>
    </footer>
  );
}
