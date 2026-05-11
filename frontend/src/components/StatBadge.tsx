'use client';

import { motion } from 'framer-motion';
import { ReactNode } from 'react';

interface StatBadgeProps {
  value: ReactNode;
  label: string;
  index?: number;
}

export function StatBadge({ value, label, index = 0 }: StatBadgeProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, delay: 0.05 * index }}
      className="relative"
    >
      <div className="absolute -top-1 -left-1 w-3 h-3 border-t-2 border-l-2 border-black dark:border-white" />
      <div className="absolute -top-1 -right-1 w-3 h-3 border-t-2 border-r-2 border-black dark:border-white" />
      <div className="absolute -bottom-1 -left-1 w-3 h-3 border-b-2 border-l-2 border-black dark:border-white" />
      <div className="absolute -bottom-1 -right-1 w-3 h-3 border-b-2 border-r-2 border-black dark:border-white" />
      <div className="border border-gray-200 dark:border-gray-800 px-4 py-3 text-center bg-white dark:bg-black">
        <div className="font-orbitron text-2xl md:text-3xl font-bold tracking-tight">
          {value}
        </div>
        <div className="text-xs tracking-widest uppercase text-gray-500 dark:text-gray-400 mt-1">
          {label}
        </div>
      </div>
    </motion.div>
  );
}
