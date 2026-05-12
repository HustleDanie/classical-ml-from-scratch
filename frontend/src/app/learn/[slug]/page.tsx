import Link from 'next/link';
import { notFound } from 'next/navigation';
import { ChevronRight } from 'lucide-react';
import {
  getLearnArticle,
  getLearnNeighbours,
  listLearnArticles,
  readLearnArticle,
} from '@/lib/content';
import { MarkdownRenderer } from '@/components/MarkdownRenderer';
import { ReadingProgress } from '@/components/ReadingProgress';
import { PhaseNav } from '@/components/PhaseNav';
import { PhaseProgress } from '@/components/PhaseProgress';

export function generateStaticParams() {
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
  const isPhase = slug.includes('_phase_');
  const displayTitle = isPhase
    ? article.title.replace(/^Phase\s+\d+:\s*/i, '')
    : article.title;

  return (
    <>
      <ReadingProgress />

      {/* Breadcrumb + progress dot strip */}
      <section className="border-b border-gray-200 dark:border-gray-800">
        <div className="max-w-7xl mx-auto px-4 md:px-6 py-4 flex flex-wrap items-center justify-between gap-3">
          <nav
            aria-label="Breadcrumb"
            className="flex items-center gap-2 text-[10px] font-mono tracking-[0.2em] uppercase text-gray-400"
          >
            <Link
              href="/learn"
              className="hover:text-emerald-700 dark:hover:text-emerald-300 transition-colors"
            >
              Learn
            </Link>
            <ChevronRight className="w-3 h-3" />
            <span className="text-black dark:text-white">{article.title}</span>
          </nav>
          {isPhase && <PhaseProgress currentSlug={slug} />}
        </div>
      </section>

      {/* Article — left rail (lg+) + body */}
      <div className="max-w-7xl mx-auto px-4 md:px-6 py-10 md:py-14">
        <div className="lg:flex lg:gap-12">
          {isPhase && <PhaseNav currentSlug={slug} />}

          <article className="flex-1 min-w-0 max-w-3xl">
            {/* Editorial article header */}
            <header className="mb-8 md:mb-10 pb-6 border-b border-gray-200 dark:border-gray-800">
              {isPhase && (
                <div className="text-[10px] font-mono tracking-[0.3em] uppercase text-emerald-700 dark:text-emerald-300 mb-3">
                  {article.trackTitle} · Phase {slug.match(/phase_(\d+)/)?.[1]?.padStart(2, '0')}
                </div>
              )}
              <h1 className="font-[family-name:var(--font-plex-serif)] text-3xl md:text-5xl font-normal leading-[1.1] tracking-tight">
                {displayTitle}
              </h1>
              {article.summary && (
                <p className="font-[family-name:var(--font-plex-serif)] mt-4 text-lg md:text-xl leading-snug text-gray-600 dark:text-gray-300 italic">
                  {article.summary}
                </p>
              )}
            </header>

            <div className="prose-mlfs max-w-none">
              <MarkdownRenderer source={body} />
            </div>

            {/* Prev / next */}
            {(prev || next || (isPhase && !prev)) && (
              <nav className="mt-16 pt-6 border-t border-gray-200 dark:border-gray-800 flex items-center justify-between gap-4">
                <div>
                  {prev ? (
                    <Link href={`/learn/${prev.slug}`} className="group block">
                      <div className="text-[10px] font-mono tracking-[0.3em] uppercase text-gray-400">
                        ← Previous
                      </div>
                      <div className="text-sm font-semibold mt-1 group-hover:text-emerald-700 dark:group-hover:text-emerald-300 transition-colors">
                        {prev.title.replace(/^Phase\s+\d+:\s*/i, '')}
                      </div>
                    </Link>
                  ) : (
                    isPhase && (
                      <Link href="/learn" className="group block">
                        <div className="text-[10px] font-mono tracking-[0.3em] uppercase text-gray-400">
                          ← Previous
                        </div>
                        <div className="text-sm font-semibold mt-1 group-hover:text-emerald-700 dark:group-hover:text-emerald-300 transition-colors">
                          The Classical ML Pipeline
                        </div>
                      </Link>
                    )
                  )}
                </div>
                <div className="text-right">
                  {next && (
                    <Link href={`/learn/${next.slug}`} className="group block">
                      <div className="text-[10px] font-mono tracking-[0.3em] uppercase text-gray-400">
                        Next →
                      </div>
                      <div className="text-sm font-semibold mt-1 group-hover:text-emerald-700 dark:group-hover:text-emerald-300 transition-colors">
                        {next.title.replace(/^Phase\s+\d+:\s*/i, '')}
                      </div>
                    </Link>
                  )}
                </div>
              </nav>
            )}
          </article>
        </div>
      </div>
    </>
  );
}
