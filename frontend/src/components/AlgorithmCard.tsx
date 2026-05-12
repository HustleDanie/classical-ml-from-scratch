'use client';

import Link from 'next/link';
import { motion } from 'framer-motion';
import { ArrowUpRight } from 'lucide-react';
import { Algorithm, CATEGORY_ACCENT } from '@/lib/algorithms';

interface AlgorithmCardProps {
  algorithm: Algorithm;
  index: number;
}

export function AlgorithmCard({ algorithm, index }: AlgorithmCardProps) {
  const accent = CATEGORY_ACCENT[algorithm.category];

  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.35, delay: Math.min(index, 8) * 0.04 }}
      className="group"
    >
      <Link
        href={`/algorithms/${algorithm.slug}`}
        className="flex h-full flex-col border border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-900 overflow-hidden hover:border-emerald-600 dark:hover:border-emerald-400 hover:-translate-y-0.5 hover:shadow-md dark:hover:shadow-emerald-950/40 transition-all duration-200"
      >
        {/* Hero plot preview */}
        <div className="relative h-44 bg-gray-50 dark:bg-gray-950 overflow-hidden">
          <div className="absolute top-3 left-3 z-10 bg-black dark:bg-white text-white dark:text-black px-2 py-1 font-mono text-xs tracking-wider">
            #{algorithm.num}
          </div>
          <div
            className={`absolute top-3 right-3 z-10 border ${accent} px-2 py-1 font-mono text-[10px] tracking-wider bg-white/80 dark:bg-black/80 backdrop-blur-sm`}
          >
            {algorithm.category.toUpperCase()}
          </div>
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={`/plots/${algorithm.folder}/${algorithm.heroPlot}`}
            alt={`${algorithm.name} — preview plot`}
            className="absolute inset-0 w-full h-full object-cover opacity-90 group-hover:opacity-100 group-hover:scale-[1.02] transition-all duration-300"
            loading="lazy"
          />
        </div>

        {/* Card body */}
        <div className="flex flex-1 flex-col p-5 gap-3">
          <div className="flex items-start justify-between gap-3">
            <h3 className="font-orbitron font-bold text-sm tracking-wide leading-snug group-hover:text-emerald-800 dark:group-hover:text-emerald-200 transition-colors">
              {algorithm.name.toUpperCase()}
            </h3>
            <ArrowUpRight className="w-4 h-4 text-gray-300 dark:text-gray-700 group-hover:text-emerald-600 dark:group-hover:text-emerald-400 group-hover:translate-x-0.5 group-hover:-translate-y-0.5 transition-all flex-shrink-0 mt-0.5" />
          </div>
          <p className="text-gray-500 dark:text-gray-400 text-xs line-clamp-2 leading-relaxed flex-1">
            {algorithm.tagline}
          </p>
          <div className="text-[10px] text-gray-400 dark:text-gray-500 font-mono tracking-widest uppercase pt-1">
            {algorithm.bestFor}
          </div>
        </div>
      </Link>
    </motion.div>
  );
}
