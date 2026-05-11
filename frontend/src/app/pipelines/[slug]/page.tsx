import Link from 'next/link';
import { notFound } from 'next/navigation';
import { ArrowLeft } from 'lucide-react';
import { PIPELINES, getPipeline } from '@/lib/pipelines';
import {
  listPipelinePlots,
  readPipelineSource,
  readPipelineReadme,
} from '@/lib/content';
import { MarkdownRenderer } from '@/components/MarkdownRenderer';
import { CodeBlock } from '@/components/CodeBlock';
import { PlotGallery } from '@/components/PlotGallery';
import { captionFromFilename } from '@/lib/captions';
import { HeroGrid } from '@/components/HeroGrid';

export function generateStaticParams() {
  return PIPELINES.map((p) => ({ slug: p.slug }));
}

export async function generateMetadata({
  params,
}: {
  params: Promise<{ slug: string }>;
}) {
  const { slug } = await params;
  const pipeline = getPipeline(slug);
  if (!pipeline) return {};
  return {
    title: `${pipeline.name} | Classical ML From Scratch`,
    description: pipeline.blurb,
  };
}

interface PageProps {
  params: Promise<{ slug: string }>;
}

export default async function PipelinePage({ params }: PageProps) {
  const { slug } = await params;
  const pipeline = getPipeline(slug);
  if (!pipeline) notFound();

  const sourceCode = readPipelineSource(pipeline.folder, pipeline.sourceFile);
  const readmeMd = readPipelineReadme(pipeline.folder);
  const plotFiles = listPipelinePlots(pipeline.folder);
  const plotBase = `/plots/pipelines/${pipeline.folder}`;
  const plots = plotFiles.map((f) => ({
    src: `${plotBase}/${f}`,
    caption: captionFromFilename(f),
  }));

  return (
    <>
      <section className="relative py-12 md:py-16 overflow-hidden border-b border-gray-200 dark:border-gray-800">
        <HeroGrid />
        <div className="relative max-w-5xl mx-auto px-4 md:px-6">
          <Link
            href="/pipelines"
            className="inline-flex items-center gap-2 text-xs tracking-widest uppercase text-gray-500 hover:text-black dark:hover:text-white transition-colors mb-6"
          >
            <ArrowLeft className="w-3 h-3" /> All pipelines
          </Link>

          <div className="text-[10px] font-mono tracking-[0.3em] uppercase text-gray-400 mb-2">
            {pipeline.task} · {pipeline.dataset}
          </div>
          <h1 className="font-orbitron text-3xl md:text-5xl font-bold tracking-tight mb-4">
            {pipeline.name.toUpperCase()}
          </h1>
          <p className="text-gray-600 dark:text-gray-300 text-base md:text-lg max-w-3xl leading-relaxed">
            {pipeline.blurb}
          </p>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mt-8">
            {pipeline.stats.map((stat) => (
              <div
                key={stat.label}
                className="relative border border-gray-200 dark:border-gray-800 px-3 py-3 bg-white dark:bg-gray-900"
              >
                <div className="absolute -top-1 -left-1 w-3 h-3 border-t-2 border-l-2 border-black dark:border-white" />
                <div className="absolute -bottom-1 -right-1 w-3 h-3 border-b-2 border-r-2 border-black dark:border-white" />
                <div className="font-orbitron text-lg font-bold">{stat.value}</div>
                <div className="text-[10px] tracking-widest uppercase text-gray-400 mt-1">
                  {stat.label}
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Skills */}
      <section className="max-w-5xl mx-auto px-4 md:px-6 py-10">
        <div className="mb-6">
          <div className="text-[10px] font-mono tracking-[0.3em] uppercase text-gray-400 mb-1">
            SECTION&nbsp;01
          </div>
          <h2 className="font-orbitron text-2xl md:text-3xl font-bold tracking-wider">
            SKILLS DEMONSTRATED
          </h2>
        </div>
        <ul className="grid grid-cols-1 md:grid-cols-2 gap-2">
          {pipeline.skills.map((skill, i) => (
            <li
              key={skill}
              className="border border-gray-200 dark:border-gray-800 px-4 py-3"
            >
              <span className="font-mono text-[10px] tracking-widest text-gray-400 mr-3">
                {String(i + 1).padStart(2, '0')}
              </span>
              <span className="text-sm">{skill}</span>
            </li>
          ))}
        </ul>
      </section>

      {/* README */}
      {readmeMd && (
        <section className="max-w-4xl mx-auto px-4 md:px-6 py-10 border-t border-gray-200 dark:border-gray-800">
          <div className="mb-6">
            <div className="text-[10px] font-mono tracking-[0.3em] uppercase text-gray-400 mb-1">
              SECTION&nbsp;02
            </div>
            <h2 className="font-orbitron text-2xl md:text-3xl font-bold tracking-wider">
              NOTES
            </h2>
          </div>
          <MarkdownRenderer source={readmeMd} imageBase={plotBase} />
        </section>
      )}

      {/* Source code */}
      {sourceCode && (
        <section className="max-w-5xl mx-auto px-4 md:px-6 py-10 border-t border-gray-200 dark:border-gray-800">
          <div className="mb-6">
            <div className="text-[10px] font-mono tracking-[0.3em] uppercase text-gray-400 mb-1">
              SECTION&nbsp;03
            </div>
            <h2 className="font-orbitron text-2xl md:text-3xl font-bold tracking-wider">
              PIPELINE SOURCE
            </h2>
            <p className="text-sm text-gray-500 dark:text-gray-400 mt-2">
              <span className="font-mono">
                real_world_practice/{pipeline.folder}/{pipeline.sourceFile}
              </span>
            </p>
          </div>
          <CodeBlock code={sourceCode} language="python" />
        </section>
      )}

      {/* Plots */}
      {plots.length > 0 && (
        <section className="max-w-7xl mx-auto px-4 md:px-6 py-10 border-t border-gray-200 dark:border-gray-800">
          <div className="mb-6 flex items-end justify-between">
            <div>
              <div className="text-[10px] font-mono tracking-[0.3em] uppercase text-gray-400 mb-1">
                SECTION&nbsp;04
              </div>
              <h2 className="font-orbitron text-2xl md:text-3xl font-bold tracking-wider">
                VISUALISATIONS
              </h2>
            </div>
            <span className="text-[10px] font-mono tracking-widest uppercase text-gray-400">
              {String(plots.length).padStart(2, '0')} plots
            </span>
          </div>
          <PlotGallery plots={plots} />
        </section>
      )}
    </>
  );
}
