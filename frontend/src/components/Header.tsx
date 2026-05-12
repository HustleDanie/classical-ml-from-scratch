'use client';

import Link from 'next/link';
import { motion } from 'framer-motion';
import { Cpu } from 'lucide-react';
import { ThemeToggle } from './ThemeToggle';

const NAV_ITEMS: { href: string; label: string; external?: boolean }[] = [
  { href: '/learn', label: 'Learn' },
  { href: '/practice', label: 'Practice' },
  { href: '/algorithms', label: 'Algorithms' },
];

export function Header() {
  return (
    <header className="sticky top-0 z-50 bg-white/80 dark:bg-black/80 backdrop-blur-md border-b border-gray-200 dark:border-gray-800">
      <div className="max-w-7xl mx-auto px-4 md:px-6 py-4">
        <div className="flex items-center justify-between gap-4">
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
          >
            <Link href="/" className="flex items-center gap-3 group">
              <div className="relative">
                <div className="w-10 h-10 border-2 border-black dark:border-white flex items-center justify-center">
                  <Cpu className="w-5 h-5" />
                </div>
                <div className="absolute -top-1 -left-1 w-3 h-3 border-t-2 border-l-2 border-black dark:border-white" />
                <div className="absolute -bottom-1 -right-1 w-3 h-3 border-b-2 border-r-2 border-black dark:border-white" />
              </div>
              <div>
                <h1 className="font-orbitron text-base md:text-xl font-bold tracking-wider whitespace-nowrap">
                  CLASSICAL&nbsp;ML
                </h1>
                <p className="text-[10px] md:text-xs text-gray-500 dark:text-gray-400 tracking-widest uppercase">
                  From Scratch
                </p>
              </div>
            </Link>
          </motion.div>

          <nav className="hidden md:flex items-center gap-6">
            {NAV_ITEMS.map((item) => (
              <Link
                key={item.href}
                href={item.href}
                target={item.external ? '_blank' : undefined}
                rel={item.external ? 'noopener noreferrer' : undefined}
                className="text-xs tracking-widest uppercase text-gray-500 dark:text-gray-400 hover:text-black dark:hover:text-white transition-colors"
              >
                {item.label}
              </Link>
            ))}
          </nav>

          <div className="flex items-center gap-3">
            <ThemeToggle />
          </div>
        </div>

        <nav className="md:hidden mt-3 flex items-center gap-4">
          {NAV_ITEMS.map((item) => (
            <Link
              key={item.href}
              href={item.href}
              target={item.external ? '_blank' : undefined}
              rel={item.external ? 'noopener noreferrer' : undefined}
              className="text-[10px] tracking-widest uppercase text-gray-500 dark:text-gray-400 hover:text-black dark:hover:text-white transition-colors"
            >
              {item.label}
            </Link>
          ))}
        </nav>
      </div>
    </header>
  );
}
