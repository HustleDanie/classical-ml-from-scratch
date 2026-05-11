'use client';

import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { X } from 'lucide-react';

interface PlotGalleryProps {
  /** Each plot's URL path under public/, e.g. "/plots/01_linear_regression/01_cost_convergence.png". */
  plots: { src: string; caption: string }[];
}

export function PlotGallery({ plots }: PlotGalleryProps) {
  const [active, setActive] = useState<number | null>(null);

  if (plots.length === 0) return null;

  return (
    <>
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
        {plots.map((plot, i) => (
          <motion.button
            key={plot.src}
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3, delay: i * 0.03 }}
            type="button"
            onClick={() => setActive(i)}
            className="group relative bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-800 overflow-hidden card-hover text-left"
          >
            <div className="absolute -top-1 -left-1 w-3 h-3 border-t-2 border-l-2 border-transparent group-hover:border-black dark:group-hover:border-white transition-colors duration-300" />
            <div className="absolute -top-1 -right-1 w-3 h-3 border-t-2 border-r-2 border-transparent group-hover:border-black dark:group-hover:border-white transition-colors duration-300" />
            <div className="absolute -bottom-1 -left-1 w-3 h-3 border-b-2 border-l-2 border-transparent group-hover:border-black dark:group-hover:border-white transition-colors duration-300" />
            <div className="absolute -bottom-1 -right-1 w-3 h-3 border-b-2 border-r-2 border-transparent group-hover:border-black dark:group-hover:border-white transition-colors duration-300" />
            <div className="aspect-[4/3] bg-gray-50 dark:bg-gray-950 flex items-center justify-center overflow-hidden">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={plot.src}
                alt={plot.caption}
                className="w-full h-full object-contain"
                loading="lazy"
              />
            </div>
            <div className="px-3 py-2 text-[11px] tracking-wider uppercase text-gray-500 dark:text-gray-400 border-t border-gray-100 dark:border-gray-800">
              {plot.caption}
            </div>
          </motion.button>
        ))}
      </div>

      <AnimatePresence>
        {active !== null && plots[active] && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-[100] bg-black/90 backdrop-blur-sm flex items-center justify-center p-4 md:p-8"
            onClick={() => setActive(null)}
          >
            <button
              onClick={() => setActive(null)}
              className="absolute top-4 right-4 w-10 h-10 border border-white text-white flex items-center justify-center hover:bg-white hover:text-black transition-colors"
              aria-label="Close"
            >
              <X className="w-5 h-5" />
            </button>
            <motion.div
              initial={{ scale: 0.95 }}
              animate={{ scale: 1 }}
              exit={{ scale: 0.95 }}
              className="max-w-6xl max-h-full w-full"
              onClick={(e) => e.stopPropagation()}
            >
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={plots[active].src}
                alt={plots[active].caption}
                className="w-full max-h-[80vh] object-contain bg-white"
              />
              <div className="text-center text-white text-xs tracking-widest uppercase mt-4">
                {plots[active].caption}
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
}

