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
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.35, delay: Math.min(index, 8) * 0.05 }}
      className="group"
    >
      <Link
        href={`/pipelines/${pipeline.slug}`}
        className="flex h-full flex-col border border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-900 p-6 hover:border-emerald-600 dark:hover:border-emerald-400 hover:bg-emerald-50/50 dark:hover:bg-emerald-950/20 hover:-translate-y-0.5 hover:shadow-md dark:hover:shadow-emerald-950/40 transition-all duration-200"
      >
        <div className="flex items-start justify-between gap-3 mb-4">
          <div className="w-10 h-10 border border-black dark:border-white flex items-center justify-center group-hover:border-emerald-600 dark:group-hover:border-emerald-400 transition-colors">
            <Workflow className="w-5 h-5 group-hover:text-emerald-600 dark:group-hover:text-emerald-400 transition-colors" />
          </div>
          <ArrowUpRight className="w-4 h-4 text-gray-300 dark:text-gray-700 group-hover:text-emerald-600 dark:group-hover:text-emerald-400 group-hover:translate-x-0.5 group-hover:-translate-y-0.5 transition-all" />
        </div>

        <div className="text-[10px] font-mono tracking-widest uppercase text-gray-400 mb-1">
          {pipeline.task}
        </div>
        <h3 className="font-orbitron font-bold text-lg tracking-wide mb-2 leading-snug group-hover:text-emerald-800 dark:group-hover:text-emerald-200 transition-colors">
          {pipeline.name.toUpperCase()}
        </h3>
        <p className="text-xs text-gray-500 dark:text-gray-400 mb-4">{pipeline.dataset}</p>

        <p className="text-sm text-gray-600 dark:text-gray-300 leading-relaxed mb-5 flex-1">
          {pipeline.tagline}
        </p>

        <div className="grid grid-cols-2 gap-3 pt-4 border-t border-gray-200 dark:border-gray-800">
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
