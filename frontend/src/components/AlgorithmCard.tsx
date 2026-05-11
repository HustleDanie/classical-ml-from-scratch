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
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, delay: index * 0.04 }}
      className="group relative"
    >
      <div className="absolute -top-1 -left-1 w-4 h-4 border-t-2 border-l-2 border-transparent group-hover:border-black dark:group-hover:border-white transition-colors duration-300" />
      <div className="absolute -top-1 -right-1 w-4 h-4 border-t-2 border-r-2 border-transparent group-hover:border-black dark:group-hover:border-white transition-colors duration-300" />
      <div className="absolute -bottom-1 -left-1 w-4 h-4 border-b-2 border-l-2 border-transparent group-hover:border-black dark:group-hover:border-white transition-colors duration-300" />
      <div className="absolute -bottom-1 -right-1 w-4 h-4 border-b-2 border-r-2 border-transparent group-hover:border-black dark:group-hover:border-white transition-colors duration-300" />

      <Link
        href={`/algorithms/${algorithm.slug}`}
        className="block bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-800 overflow-hidden card-hover"
      >
        <div className="relative h-44 bg-gray-50 dark:bg-gray-950 overflow-hidden">
          <div className="absolute top-3 left-3 bg-black dark:bg-white text-white dark:text-black px-2 py-1 font-mono text-xs tracking-wider z-10">
            #{algorithm.num}
          </div>
          <div
            className={`absolute top-3 right-3 border ${accent} px-2 py-1 font-mono text-[10px] tracking-wider bg-white/80 dark:bg-black/80 backdrop-blur-sm z-10`}
          >
            {algorithm.category.toUpperCase()}
          </div>
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={`/plots/${algorithm.folder}/${algorithm.heroPlot}`}
            alt={`${algorithm.name} — preview plot`}
            className="absolute inset-0 w-full h-full object-cover opacity-80 group-hover:opacity-100 transition-opacity"
            loading="lazy"
          />
          <div className="absolute inset-0 glitch-overlay pointer-events-none" />
          <div className="absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity">
            <div
              className="absolute inset-0 bg-gradient-to-b from-transparent via-black/5 to-transparent"
              style={{ backgroundSize: '100% 4px' }}
            />
          </div>
        </div>

        <div className="p-4 space-y-3">
          <div className="flex items-start justify-between gap-3">
            <h3 className="font-orbitron font-bold text-sm tracking-wide leading-snug">
              {algorithm.name.toUpperCase()}
            </h3>
            <ArrowUpRight className="w-4 h-4 text-gray-400 group-hover:text-black dark:group-hover:text-white transition-colors flex-shrink-0 mt-0.5" />
          </div>
          <p className="text-gray-500 dark:text-gray-400 text-xs line-clamp-2 leading-relaxed">
            {algorithm.tagline}
          </p>
          <div className="pt-2 text-[10px] text-gray-400 font-mono tracking-widest uppercase">
            {algorithm.bestFor}
          </div>
        </div>
      </Link>
    </motion.div>
  );
}
