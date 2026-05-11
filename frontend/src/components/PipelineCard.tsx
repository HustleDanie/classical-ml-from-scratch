'use client';

import Link from 'next/link';
import { motion } from 'framer-motion';
import { ArrowUpRight, Workflow } from 'lucide-react';
import { Pipeline } from '@/lib/pipelines';

interface PipelineCardProps {
  pipeline: Pipeline;
  index: number;
}

export function PipelineCard({ pipeline, index }: PipelineCardProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, delay: index * 0.05 }}
      className="group relative"
    >
      <div className="absolute -top-1 -left-1 w-4 h-4 border-t-2 border-l-2 border-transparent group-hover:border-black dark:group-hover:border-white transition-colors duration-300" />
      <div className="absolute -top-1 -right-1 w-4 h-4 border-t-2 border-r-2 border-transparent group-hover:border-black dark:group-hover:border-white transition-colors duration-300" />
      <div className="absolute -bottom-1 -left-1 w-4 h-4 border-b-2 border-l-2 border-transparent group-hover:border-black dark:group-hover:border-white transition-colors duration-300" />
      <div className="absolute -bottom-1 -right-1 w-4 h-4 border-b-2 border-r-2 border-transparent group-hover:border-black dark:group-hover:border-white transition-colors duration-300" />

      <Link
        href={`/pipelines/${pipeline.slug}`}
        className="block bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-800 p-6 card-hover h-full"
      >
        <div className="flex items-start justify-between gap-3 mb-4">
          <div className="w-10 h-10 border border-black dark:border-white flex items-center justify-center">
            <Workflow className="w-5 h-5" />
          </div>
          <ArrowUpRight className="w-4 h-4 text-gray-400 group-hover:text-black dark:group-hover:text-white transition-colors" />
        </div>

        <div className="text-[10px] font-mono tracking-widest uppercase text-gray-400 mb-1">
          {pipeline.task}
        </div>
        <h3 className="font-orbitron font-bold text-lg tracking-wide mb-2 leading-snug">
          {pipeline.name.toUpperCase()}
        </h3>
        <p className="text-xs text-gray-500 dark:text-gray-400 mb-4">{pipeline.dataset}</p>

        <p className="text-sm text-gray-600 dark:text-gray-300 leading-relaxed mb-5">
          {pipeline.tagline}
        </p>

        <div className="grid grid-cols-2 gap-2 pt-4 border-t border-gray-100 dark:border-gray-800">
          {pipeline.stats.slice(0, 4).map((stat) => (
            <div key={stat.label}>
              <div className="font-orbitron text-base font-bold">{stat.value}</div>
              <div className="text-[10px] tracking-widest uppercase text-gray-400">
                {stat.label}
              </div>
            </div>
          ))}
        </div>
      </Link>
    </motion.div>
  );
}
