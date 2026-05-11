import Link from 'next/link';
import { notFound } from 'next/navigation';
import { ArrowLeft, ArrowRight } from 'lucide-react';
import {
  ALGORITHMS,
  CATEGORY_ACCENT,
  getAlgorithm,
  getNeighbours,
} from '@/lib/algorithms';
import {
  listAlgorithmPlots,
  readAlgorithmReadme,
  readAlgorithmSource,
  readHowItWorks,
} from '@/lib/content';
import { MarkdownRenderer } from '@/components/MarkdownRenderer';
import { CodeBlock } from '@/components/CodeBlock';
import { PlotGallery } from '@/components/PlotGallery';
import { captionFromFilename } from '@/lib/captions';
import { HeroGrid } from '@/components/HeroGrid';
import { KeyFormula } from '@/components/KeyFormula';
import { SectionStripNav, type SectionLink } from '@/components/SectionStripNav';
import { CurriculumTopic } from '@/components/CurriculumTopic';
import { YouTubePlaylistEmbed } from '@/components/YouTubePlaylistEmbed';
import { PaperCard } from '@/components/PaperCard';
import { getLearning } from '@/lib/learning';

export function generateStaticParams() {
  return ALGORITHMS.map((a) => ({ slug: a.slug }));
}

export async function generateMetadata({
  params,
}: {
  params: Promise<{ slug: string }>;
}) {
  const { slug } = await params;
  const algo = getAlgorithm(slug);
  if (!algo) return {};
  return {
    title: `${algo.name} | Classical ML From Scratch`,
    description: algo.blurb,
  };
}

interface PageProps {
  params: Promise<{ slug: string }>;
}

export default async function AlgorithmPage({ params }: PageProps) {
  const { slug } = await params;
  const algo = getAlgorithm(slug);
  if (!algo) notFound();

  const howItWorksMd = readHowItWorks(algo.howItWorks);
  const readmeMd = readAlgorithmReadme(algo.folder);
  const sourceCode = readAlgorithmSource(algo.folder, algo.sourceFile);
  const plotFiles = listAlgorithmPlots(algo.folder);
  const { prev, next } = getNeighbours(slug);
  const accent = CATEGORY_ACCENT[algo.category];
  const plotBase = `/plots/${algo.folder}`;
  const plots = plotFiles.map((f) => ({
    src: `${plotBase}/${f}`,
    caption: captionFromFilename(f),
  }));

  const learning = getLearning(slug);
  const hasCurriculum = !!learning && learning.curriculum.length > 0;
  const hasPlaylists = !!learning && learning.playlists.length > 0;
  const hasPapers = !!learning && learning.papers.length > 0;

  const sections: SectionLink[] = [{ id: 'overview', label: 'Overview' }];
  if (hasCurriculum || hasPlaylists) {
    sections.push({ id: 'curriculum', label: 'Scheme of Work' });
  }
  if (hasPapers) {
    sections.push({ id: 'papers', label: 'Papers' });
  }

  return (
    <>
      {/* Hero */}
      <section className="relative py-12 md:py-16 overflow-hidden border-b border-gray-200 dark:border-gray-800">
        <HeroGrid />
        <div className="relative max-w-5xl mx-auto px-4 md:px-6">
          <Link
            href="/algorithms"
            className="inline-flex items-center gap-2 text-xs tracking-widest uppercase text-gray-500 hover:text-black dark:hover:text-white transition-colors mb-6"
          >
            <ArrowLeft className="w-3 h-3" /> All algorithms
          </Link>

          <div className="flex items-center gap-3 mb-3 flex-wrap">
            <span className="bg-black dark:bg-white text-white dark:text-black px-2 py-1 font-mono text-xs tracking-wider">
              #{algo.num}
            </span>
            <span className={`border ${accent} px-2 py-1 font-mono text-[10px] tracking-wider`}>
              {algo.category.toUpperCase()}
            </span>
            <span className="text-[10px] font-mono tracking-widest uppercase text-gray-400">
              {algo.bestFor}
            </span>
          </div>

          <h1 className="font-orbitron text-3xl md:text-5xl font-bold tracking-tight mb-4">
            {algo.name.toUpperCase()}
          </h1>
          <p className="text-gray-600 dark:text-gray-300 text-base md:text-lg leading-relaxed max-w-3xl">
            {algo.blurb}
          </p>

          {algo.keyFormula && (
            <div className="mt-8">
              <div className="text-[10px] font-mono tracking-[0.3em] uppercase text-gray-400 mb-2">
                Key Formula
              </div>
              <KeyFormula tex={algo.keyFormula} />
            </div>
          )}
        </div>
      </section>

      <SectionStripNav sections={sections} accent={accent} />

      {/* OVERVIEW */}
      <section id="overview" className="scroll-mt-20">
        {/* Hero plot */}
        <div className="max-w-5xl mx-auto px-4 md:px-6 py-10">
          <div className="text-[10px] font-mono tracking-[0.3em] uppercase text-gray-400 mb-3">
            SECTION&nbsp;01 / OVERVIEW
          </div>
          <div className="relative border border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-900 overflow-hidden">
            <div className="absolute -top-1 -left-1 w-4 h-4 border-t-2 border-l-2 border-black dark:border-white" />
            <div className="absolute -top-1 -right-1 w-4 h-4 border-t-2 border-r-2 border-black dark:border-white" />
            <div className="absolute -bottom-1 -left-1 w-4 h-4 border-b-2 border-l-2 border-black dark:border-white" />
            <div className="absolute -bottom-1 -right-1 w-4 h-4 border-b-2 border-r-2 border-black dark:border-white" />
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={`${plotBase}/${algo.heroPlot}`}
              alt={`${algo.name} hero plot`}
              className="w-full h-auto block"
            />
          </div>
        </div>

        {/* How It Works */}
        {howItWorksMd && (
          <div className="max-w-4xl mx-auto px-4 md:px-6 py-10 border-t border-gray-200 dark:border-gray-800">
            <h2 className="font-orbitron text-2xl md:text-3xl font-bold tracking-wider mb-6">
              HOW IT WORKS
            </h2>
            <MarkdownRenderer source={howItWorksMd} imageBase={plotBase} />
          </div>
        )}

        {/* Implementation source */}
        {sourceCode && (
          <div className="max-w-5xl mx-auto px-4 md:px-6 py-10 border-t border-gray-200 dark:border-gray-800">
            <div className="mb-6">
              <h2 className="font-orbitron text-2xl md:text-3xl font-bold tracking-wider">
                IMPLEMENTATION
              </h2>
              <p className="text-sm text-gray-500 dark:text-gray-400 mt-2">
                <span className="font-mono">algorithms/{algo.folder}/{algo.sourceFile}</span>
              </p>
            </div>
            <CodeBlock code={sourceCode} language="python" />
          </div>
        )}

        {/* Plot Gallery */}
        {plots.length > 0 && (
          <div className="max-w-7xl mx-auto px-4 md:px-6 py-10 border-t border-gray-200 dark:border-gray-800">
            <div className="mb-6 flex items-end justify-between">
              <h2 className="font-orbitron text-2xl md:text-3xl font-bold tracking-wider">
                VISUALISATIONS
              </h2>
              <span className="text-[10px] font-mono tracking-widest uppercase text-gray-400">
                {String(plots.length).padStart(2, '0')} plots
              </span>
            </div>
            <PlotGallery plots={plots} />
          </div>
        )}

        {/* README notes */}
        {readmeMd && (
          <div className="max-w-4xl mx-auto px-4 md:px-6 py-10 border-t border-gray-200 dark:border-gray-800">
            <h2 className="font-orbitron text-2xl md:text-3xl font-bold tracking-wider mb-6">
              NOTES
            </h2>
            <MarkdownRenderer source={readmeMd} imageBase={plotBase} />
          </div>
        )}
      </section>

      {/* CURRICULUM */}
      {(hasCurriculum || hasPlaylists) && learning && (
        <section
          id="curriculum"
          className="scroll-mt-20 max-w-4xl mx-auto px-4 md:px-6 py-10 border-t border-gray-200 dark:border-gray-800"
        >
          <div className="mb-8">
            <div className="text-[10px] font-mono tracking-[0.3em] uppercase text-gray-400 mb-1">
              SECTION&nbsp;02
            </div>
            <h2 className="font-orbitron text-2xl md:text-3xl font-bold tracking-wider">
              SCHEME OF WORK
            </h2>
            <p className="text-sm text-gray-500 dark:text-gray-400 mt-3 max-w-2xl leading-relaxed">
              A guided learning path. Each topic pairs a short explanation with a hand-picked
              video so you can build a complete mental model of the algorithm.
            </p>
          </div>

          {hasCurriculum && (
            <div className="grid grid-cols-1 gap-6">
              {learning.curriculum.map((t, i) => (
                <CurriculumTopic key={`${t.youtubeId}-${i}`} index={i} topic={t} />
              ))}
            </div>
          )}

          {hasPlaylists && (
            <div className="mt-12">
              <h3 className="font-orbitron text-lg md:text-xl font-bold tracking-wider mb-2">
                FURTHER PLAYLISTS
              </h3>
              <p className="text-sm text-gray-500 dark:text-gray-400 mb-6">
                Full playlists for deeper study.
              </p>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {learning.playlists.map((p) => (
                  <div key={p.playlistId}>
                    <YouTubePlaylistEmbed playlistId={p.playlistId} title={p.title} />
                    <div className="mt-3">
                      <div className="font-orbitron text-sm font-bold tracking-wide">
                        {p.title.toUpperCase()}
                      </div>
                      <div className="text-[10px] font-mono tracking-widest uppercase text-gray-400 mt-1">
                        {p.channel}
                      </div>
                      {p.blurb && (
                        <p className="text-sm text-gray-600 dark:text-gray-300 mt-2 leading-relaxed">
                          {p.blurb}
                        </p>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </section>
      )}

      {/* PAPERS */}
      {hasPapers && learning && (
        <section
          id="papers"
          className="scroll-mt-20 max-w-4xl mx-auto px-4 md:px-6 py-10 border-t border-gray-200 dark:border-gray-800"
        >
          <div className="mb-8">
            <div className="text-[10px] font-mono tracking-[0.3em] uppercase text-gray-400 mb-1">
              SECTION&nbsp;03
            </div>
            <h2 className="font-orbitron text-2xl md:text-3xl font-bold tracking-wider">
              PAPERS
            </h2>
            <p className="text-sm text-gray-500 dark:text-gray-400 mt-3 max-w-2xl leading-relaxed">
              The original and most-cited papers behind this algorithm — go straight to the source.
            </p>
          </div>
          <div className="grid grid-cols-1 gap-4">
            {learning.papers.map((p, i) => (
              <PaperCard key={`${p.title}-${i}`} paper={p} />
            ))}
          </div>
        </section>
      )}

      {/* Prev/Next */}
      <section className="max-w-5xl mx-auto px-4 md:px-6 py-10 border-t border-gray-200 dark:border-gray-800">
        <div className="grid grid-cols-2 gap-4">
          <div>
            {prev && (
              <Link
                href={`/algorithms/${prev.slug}`}
                className="group block border border-gray-200 dark:border-gray-800 p-4 hover:border-black dark:hover:border-white transition-colors"
              >
                <div className="text-[10px] font-mono tracking-widest uppercase text-gray-400 mb-1 flex items-center gap-2">
                  <ArrowLeft className="w-3 h-3" /> Previous
                </div>
                <div className="font-orbitron font-bold tracking-wide">
                  {prev.name.toUpperCase()}
                </div>
              </Link>
            )}
          </div>
          <div>
            {next && (
              <Link
                href={`/algorithms/${next.slug}`}
                className="group block border border-gray-200 dark:border-gray-800 p-4 hover:border-black dark:hover:border-white transition-colors text-right"
              >
                <div className="text-[10px] font-mono tracking-widest uppercase text-gray-400 mb-1 flex items-center gap-2 justify-end">
                  Next <ArrowRight className="w-3 h-3" />
                </div>
                <div className="font-orbitron font-bold tracking-wide">
                  {next.name.toUpperCase()}
                </div>
              </Link>
            )}
          </div>
        </div>
      </section>
    </>
  );
}
