'use client';

import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import { CodeBlock } from './CodeBlock';

interface MarkdownRendererProps {
  source: string;
  /** Prefix to prepend to relative image paths (e.g. "/plots/01_linear_regression"). */
  imageBase?: string;
}

export function MarkdownRenderer({ source, imageBase }: MarkdownRendererProps) {
  return (
    <div className="prose-mlfs">
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkMath]}
        rehypePlugins={[
          [
            rehypeKatex,
            {
              // Quiet KaTeX's noisy LaTeX-strict warnings (e.g., `%` in
              // math contexts) and don't fail the render if a single
              // expression is malformed — show the source instead.
              strict: 'ignore',
              throwOnError: false,
            },
          ],
        ]}
        components={{
          code({ className, children, ...rest }) {
            const inline = !(className && /language-/.test(className));
            const match = /language-(\w+)/.exec(className || '');
            const lang = match ? match[1] : 'text';
            const text = String(children).replace(/\n$/, '');
            if (inline) {
              return (
                <code className={className} {...rest}>
                  {children}
                </code>
              );
            }
            return <CodeBlock code={text} language={lang} />;
          },
          img({ src, alt, ...rest }) {
            let resolved = src;
            if (typeof src === 'string' && imageBase && !/^https?:\/\//.test(src) && !src.startsWith('/')) {
              resolved = `${imageBase.replace(/\/$/, '')}/${src.replace(/^\.\//, '')}`;
            }
            // eslint-disable-next-line @next/next/no-img-element
            return <img src={resolved} alt={alt ?? ''} {...rest} />;
          },
        }}
      >
        {source}
      </ReactMarkdown>
    </div>
  );
}
