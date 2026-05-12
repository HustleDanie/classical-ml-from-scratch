import Link from 'next/link';
import { notFound } from 'next/navigation';
import { ArrowLeft } from 'lucide-react';
import {
  getLearnArticle,
  getLearnNeighbours,
  listLearnArticles,
  readLearnArticle,
} from '@/lib/content';
import { MarkdownRenderer } from '@/components/MarkdownRenderer';

export function generateStaticParams() {
  // Only prerender slugs whose files actually exist; "coming soon" stubs
  // remain unreachable until their .md file is added.
  return listLearnArticles()
    .filter((a) => a.exists)
    .map((a) => ({ slug: a.slug }));
}

export async function generateMetadata({
  params,
}: {
  params: Promise<{ slug: string }>;
}) {
  const { slug } = await params;
  const article = getLearnArticle(slug);
  if (!article || !article.exists) return {};
  return {
    title: `${article.title} | Learn — Classical ML From Scratch`,
    description: article.summary,
  };
}

interface PageProps {
  params: Promise<{ slug: string }>;
}

export default async function LearnArticlePage({ params }: PageProps) {
  const { slug } = await params;
  const article = getLearnArticle(slug);
  if (!article || !article.exists) notFound();
  const body = readLearnArticle(slug);
  if (!body) notFound();

  const { prev, next } = getLearnNeighbours(slug);

  // Phase pages chain back to the pipeline guide at /learn when there's no
  // previous article in the index.
  const isPhase = slug.includes('_phase_');
  const showPipelineBack = !prev && isPhase;

  return (
    <>
      <section className="relative py-12 md:py-16 overflow-hidden border-b border-gray-200 dark:border-gray-800">
        <div className="relative max-w-4xl mx-auto px-4 md:px-6">
          <Link
            href="/learn"
            className="inline-flex items-center gap-2 text-xs tracking-widest uppercase text-gray-500 hover:text-black dark:hover:text-white transition-colors mb-6"
          >
            <ArrowLeft className="w-3 h-3" /> All learn articles
          </Link>

          <div className="text-[10px] font-mono tracking-[0.3em] uppercase mb-2 text-gray-400">
            Learn <span className="text-gray-300 dark:text-gray-600">·</span>{' '}
            <span className="text-black dark:text-white">{article.trackTitle}</span>
          </div>
          <h1 className="font-orbitron text-2xl md:text-4xl font-bold tracking-tight">
            {article.title}
          </h1>
          {article.summary && (
            <p className="mt-3 text-sm md:text-base text-gray-600 dark:text-gray-300 max-w-3xl">
              {article.summary}
            </p>
          )}
        </div>
      </section>

      <section className="max-w-4xl mx-auto px-4 md:px-6 py-10">
        <article className="prose prose-zinc dark:prose-invert max-w-none">
          <MarkdownRenderer source={body} />
        </article>
      </section>

      {(prev || next || showPipelineBack) && (
        <nav className="max-w-4xl mx-auto px-4 md:px-6 pb-16 flex items-center justify-between gap-4 border-t border-gray-200 dark:border-gray-800 pt-6">
          <div>
            {prev && (
              <Link href={`/learn/${prev.slug}`} className="group block">
                <div className="text-[10px] font-mono tracking-[0.3em] uppercase text-gray-400">
                  ← Previous
                </div>
                <div className="text-sm font-semibold mt-1 group-hover:underline">
                  {prev.title}
                </div>
              </Link>
            )}
            {!prev && showPipelineBack && (
              <Link href="/learn" className="group block">
                <div className="text-[10px] font-mono tracking-[0.3em] uppercase text-gray-400">
                  ← Previous
                </div>
                <div className="text-sm font-semibold mt-1 group-hover:underline">
                  The Classical ML Pipeline
                </div>
              </Link>
            )}
          </div>
          <div className="text-right">
            {next && (
              <Link href={`/learn/${next.slug}`} className="group block">
                <div className="text-[10px] font-mono tracking-[0.3em] uppercase text-gray-400">
                  Next →
                </div>
                <div className="text-sm font-semibold mt-1 group-hover:underline">
                  {next.title}
                </div>
              </Link>
            )}
          </div>
        </nav>
      )}
    </>
  );
}
