'use client';

import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus, oneLight } from 'react-syntax-highlighter/dist/esm/styles/prism';
import { useTheme } from 'next-themes';
import { useEffect, useState } from 'react';

interface CodeBlockProps {
  code: string;
  language?: string;
  caption?: string;
}

export function CodeBlock({ code, language = 'python', caption }: CodeBlockProps) {
  const { resolvedTheme } = useTheme();
  const [mounted, setMounted] = useState(false);
  useEffect(() => {
    // SSR-hydration guard: pick the syntax-highlight theme only after mount.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setMounted(true);
  }, []);

  const style = mounted && resolvedTheme === 'dark' ? vscDarkPlus : oneLight;

  return (
    <div className="my-4">
      {caption && (
        <div className="text-[10px] tracking-widest uppercase text-gray-400 mb-2">
          {caption}
        </div>
      )}
      <div className="relative border border-gray-200 dark:border-gray-800 overflow-hidden">
        <div className="absolute top-2 right-3 text-[10px] tracking-widest uppercase text-gray-400 font-mono z-10">
          {language}
        </div>
        <SyntaxHighlighter
          language={language}
          style={style}
          customStyle={{
            margin: 0,
            padding: '1rem 1rem 1rem 1rem',
            background: 'transparent',
            fontSize: '0.8rem',
            lineHeight: 1.55,
            fontFamily:
              'var(--font-plex-mono), ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace',
          }}
          codeTagProps={{
            style: {
              fontFamily:
                'var(--font-plex-mono), ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace',
            },
          }}
          showLineNumbers={false}
        >
          {code}
        </SyntaxHighlighter>
      </div>
    </div>
  );
}
