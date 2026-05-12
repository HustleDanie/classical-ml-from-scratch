'use client';

import { useEffect, useState } from 'react';

/**
 * Thin emerald progress bar pinned to the top of the viewport that fills
 * as the user scrolls through the article. Pure scroll math — no scroll
 * libraries, no observers — so it's cheap.
 */
export function ReadingProgress() {
  const [progress, setProgress] = useState(0);

  useEffect(() => {
    function onScroll() {
      const docHeight =
        document.documentElement.scrollHeight - window.innerHeight;
      if (docHeight <= 0) {
        setProgress(0);
        return;
      }
      const pct = (window.scrollY / docHeight) * 100;
      setProgress(Math.max(0, Math.min(100, pct)));
    }
    onScroll(); // initial paint
    window.addEventListener('scroll', onScroll, { passive: true });
    window.addEventListener('resize', onScroll);
    return () => {
      window.removeEventListener('scroll', onScroll);
      window.removeEventListener('resize', onScroll);
    };
  }, []);

  return (
    <div
      aria-hidden="true"
      className="fixed top-0 left-0 right-0 z-[60] h-0.5 bg-transparent pointer-events-none"
    >
      <div
        className="h-full bg-emerald-500 dark:bg-emerald-400 transition-[width] duration-75 ease-out"
        style={{ width: `${progress}%` }}
      />
    </div>
  );
}
