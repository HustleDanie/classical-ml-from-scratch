'use client';

import { motion } from 'framer-motion';
import { Algorithm } from '@/lib/algorithms';
import { AlgorithmCard } from './AlgorithmCard';

interface AlgorithmGridProps {
  algorithms: Algorithm[];
}

export function AlgorithmGrid({ algorithms }: AlgorithmGridProps) {
  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6"
    >
      {algorithms.map((algorithm, index) => (
        <AlgorithmCard key={algorithm.slug} algorithm={algorithm} index={index} />
      ))}
    </motion.div>
  );
}
