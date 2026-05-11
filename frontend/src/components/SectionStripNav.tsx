'use client';

import { useEffect, useState } from 'react';

export interface SectionLink {
  id: string;
  label: string;
}

interface SectionStripNavProps {
  sections: SectionLink[];
  /** Tailwind border + text class for the active pill, e.g. "border-blue-500 text-blue-500". */
  accent: string;
}

export function SectionStripNav({ sections, accent }: SectionStripNavProps) {
  const [active, setActive] = useState<string>(sections[0]?.id ?? '');

  useEffect(() => {
    if (sections.length === 0) return;
    const elements = sections
      .map((s) => document.getElementById(s.id))
      .filter((el): el is HTMLElement => el !== null);
    if (elements.length === 0) return;

    const observer = new IntersectionObserver(
      (entries) => {
        const visible = entries
          .filter((e) => e.isIntersecting)
          .sort((a, b) => b.intersectionRatio - a.intersectionRatio);
        if (visible.length > 0) {
          setActive(visible[0].target.id);
        }
      },
      {
        rootMargin: '-40% 0px -50% 0px',
        threshold: [0, 0.25, 0.5, 0.75, 1],
      },
    );

    elements.forEach((el) => observer.observe(el));
    return () => observer.disconnect();
  }, [sections]);

  function handleClick(e: React.MouseEvent<HTMLAnchorElement>, id: string) {
    e.preventDefault();
    const el = document.getElementById(id);
    if (!el) return;
    el.scrollIntoView({ behavior: 'smooth', block: 'start' });
    history.replaceState(null, '', `#${id}`);
    setActive(id);
  }

  return (
    <nav
      aria-label="Page sections"
      className="sticky top-0 z-30 border-y border-gray-200 dark:border-gray-800 bg-white/85 dark:bg-black/85 backdrop-blur"
    >
      <div className="max-w-5xl mx-auto px-4 md:px-6 py-3 flex flex-wrap items-center gap-2 md:gap-3">
        {sections.map((s) => {
          const isActive = active === s.id;
          const cls = isActive
            ? `border ${accent} px-3 py-1 font-mono text-[10px] md:text-xs tracking-widest`
            : 'border border-gray-300 dark:border-gray-700 text-gray-500 hover:text-black dark:hover:text-white hover:border-black dark:hover:border-white px-3 py-1 font-mono text-[10px] md:text-xs tracking-widest transition-colors';
          return (
            <a key={s.id} href={`#${s.id}`} onClick={(e) => handleClick(e, s.id)} className={cls}>
              {s.label.toUpperCase()}
            </a>
          );
        })}
      </div>
    </nav>
  );
}
