import Link from 'next/link';
import { notFound } from 'next/navigation';
import { ArrowLeft } from 'lucide-react';
import {
  listExpertScenarios,
  readScenario,
  readScenarioCatalog,
  readScenarioMethodology,
  type ScenarioType,
} from '@/lib/content';
import { MarkdownRenderer } from '@/components/MarkdownRenderer';

const TYPE_LABEL: Record<ScenarioType, { label: string; accent: string }> = {
  classification: { label: 'Classification', accent: 'text-emerald-500' },
  regression: { label: 'Regression', accent: 'text-blue-500' },
};

export function generateStaticParams() {
  const dynamic = listExpertScenarios().map((s) => ({ type: s.type, slug: s.slug }));
  const special: { type: string; slug: string }[] = [];
  for (const t of ['classification', 'regression']) {
    special.push({ type: t, slug: 'CATALOG' });
    special.push({ type: t, slug: '_methodology' });
  }
  return [...dynamic, ...special];
}

export async function generateMetadata({
  params,
}: {
  params: Promise<{ type: string; slug: string }>;
}) {
  const { type, slug } = await params;
  const scenarios = listExpertScenarios();
  const s = scenarios.find((x) => x.type === type && x.slug === slug);
  if (s) {
    return { title: `${s.title} | Classical ML From Scratch`, description: s.summary };
  }
  if (slug === 'CATALOG') {
    return { title: `${type} Catalog | Classical ML From Scratch` };
  }
  if (slug === '_methodology') {
    return { title: `${type} Methodology | Classical ML From Scratch` };
  }
  return {};
}

interface PageProps {
  params: Promise<{ type: string; slug: string }>;
}

function isValidType(t: string): t is ScenarioType {
  return t === 'classification' || t === 'regression';
}

export default async function ScenarioDetailPage({ params }: PageProps) {
  const { type, slug } = await params;
  if (!isValidType(type)) notFound();

  const meta = TYPE_LABEL[type];

  let title: string;
  let body: string | null;
  let isSpecial = false;

  if (slug === 'CATALOG') {
    title = `${meta.label} Catalog`;
    body = readScenarioCatalog(type);
    isSpecial = true;
  } else if (slug === '_methodology') {
    title = `${meta.label} — From Brief to Solution`;
    body = readScenarioMethodology(type);
    isSpecial = true;
  } else {
    const scenario = listExpertScenarios().find(
      (s) => s.type === type && s.slug === slug,
    );
    if (!scenario) notFound();
    title = scenario.title;
    body = readScenario(type, slug);
  }

  if (!body) notFound();

  // Find prev/next within the same type
  const allInType = listExpertScenarios().filter((s) => s.type === type);
  const idx = allInType.findIndex((s) => s.slug === slug);
  const prev = !isSpecial && idx > 0 ? allInType[idx - 1] : null;
  const next = !isSpecial && idx >= 0 && idx < allInType.length - 1 ? allInType[idx + 1] : null;

  return (
    <>
      <section className="relative py-12 md:py-16 overflow-hidden border-b border-gray-200 dark:border-gray-800">
        <div className="relative max-w-4xl mx-auto px-4 md:px-6">
          <Link
            href="/scenarios"
            className="inline-flex items-center gap-2 text-xs tracking-widest uppercase text-gray-500 hover:text-black dark:hover:text-white transition-colors mb-6"
          >
            <ArrowLeft className="w-3 h-3" /> All scenarios
          </Link>

          <div
            className={`text-[10px] font-mono tracking-[0.3em] uppercase mb-2 ${meta.accent}`}
          >
            {meta.label}
            {!isSpecial && (
              <>
                {' '}
                <span className="text-gray-400">·</span>{' '}
                <Link
                  href={`/scenarios/${type}/CATALOG`}
                  className="hover:underline"
                >
                  catalog
                </Link>{' '}
                <span className="text-gray-400">·</span>{' '}
                <Link
                  href={`/scenarios/${type}/_methodology`}
                  className="hover:underline"
                >
                  methodology
                </Link>
              </>
            )}
          </div>
          <h1 className="font-orbitron text-2xl md:text-4xl font-bold tracking-tight">
            {title}
          </h1>
        </div>
      </section>

      <section className="max-w-4xl mx-auto px-4 md:px-6 py-10">
        <article className="prose prose-zinc dark:prose-invert max-w-none">
          <MarkdownRenderer source={body} />
        </article>
      </section>

      {(prev || next) && (
        <nav className="max-w-4xl mx-auto px-4 md:px-6 pb-16 flex items-center justify-between gap-4 border-t border-gray-200 dark:border-gray-800 pt-6">
          <div>
            {prev && (
              <Link
                href={`/scenarios/${prev.type}/${prev.slug}`}
                className="group block"
              >
                <div className="text-[10px] font-mono tracking-[0.3em] uppercase text-gray-400">
                  ← Previous
                </div>
                <div className="text-sm font-semibold mt-1 group-hover:underline">
                  {prev.title}
                </div>
              </Link>
            )}
          </div>
          <div className="text-right">
            {next && (
              <Link
                href={`/scenarios/${next.type}/${next.slug}`}
                className="group block"
              >
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
