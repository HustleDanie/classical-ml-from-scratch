import fs from 'node:fs';
import path from 'node:path';

const CONTENT_ROOT = path.join(process.cwd(), 'content');

function safeRead(rel: string): string | null {
  const full = path.join(CONTENT_ROOT, rel);
  try {
    return fs.readFileSync(full, 'utf8');
  } catch {
    return null;
  }
}

function safeListPlots(folder: string): string[] {
  const full = path.join(process.cwd(), 'public', 'plots', folder);
  try {
    return fs
      .readdirSync(full)
      .filter((f) => /\.(png|jpe?g|svg)$/i.test(f))
      .sort();
  } catch {
    return [];
  }
}

export function readAlgorithmReadme(folder: string): string | null {
  return safeRead(path.join('algorithms', folder, 'README.md'));
}

export function readAlgorithmSource(folder: string, sourceFile: string): string | null {
  return safeRead(path.join('algorithms', folder, sourceFile));
}

export function readHowItWorks(slug: string): string | null {
  return safeRead(path.join('how_it_works', `${slug}.md`));
}

export function readPipelineSource(folder: string, sourceFile: string): string | null {
  return safeRead(path.join('pipelines', folder, sourceFile));
}

export function readPipelineReadme(folder: string): string | null {
  return safeRead(path.join('pipelines', folder, 'README.md'));
}

export function listAlgorithmPlots(folder: string): string[] {
  return safeListPlots(folder);
}

export function listPipelinePlots(folder: string): string[] {
  const full = path.join(process.cwd(), 'public', 'plots', 'pipelines', folder);
  try {
    return fs
      .readdirSync(full)
      .filter((f) => /\.(png|jpe?g|svg)$/i.test(f))
      .sort();
  } catch {
    return [];
  }
}

export type ScenarioType = 'classification' | 'regression';

export interface Scenario {
  slug: string;
  title: string;
  type: ScenarioType;
  /** Numeric prefix from the filename (01, 02, ...) used for ordering */
  num: string;
  /** First non-blockquote paragraph from the file — used as list-page summary */
  summary: string;
  /** True for full ~500-line deep dives; false for compact reference entries */
  isDeepDive: boolean;
}

export function listExpertScenarios(): Scenario[] {
  const root = path.join(CONTENT_ROOT, 'scenarios');
  const out: Scenario[] = [];
  for (const type of ['classification', 'regression'] as ScenarioType[]) {
    const dir = path.join(root, type);
    if (!fs.existsSync(dir)) continue;
    for (const f of fs.readdirSync(dir)) {
      if (!f.endsWith('.md')) continue;
      // Skip the CATALOG.md index and methodology guide; surfaced separately.
      if (f === 'CATALOG.md' || f === '_methodology.md') continue;
      const slug = f.replace(/\.md$/, '');
      const numMatch = slug.match(/^(\d+)_/);
      const num = numMatch ? numMatch[1] : '99';
      const raw = fs.readFileSync(path.join(dir, f), 'utf8');
      const lines = raw.split('\n');
      const firstHeading = lines.find((l) => l.startsWith('# '));
      const title = firstHeading
        ? firstHeading.replace(/^#\s+/, '').trim()
        : slug.replace(/_/g, ' ');
      // Summary: first non-blockquote paragraph after the title.
      let summary = '';
      let pastTitle = false;
      for (const line of lines) {
        if (line.startsWith('# ')) {
          pastTitle = true;
          continue;
        }
        if (!pastTitle) continue;
        const t = line.trim();
        if (!t) continue;
        if (t.startsWith('>')) continue;
        if (t.startsWith('---')) continue;
        if (t.startsWith('#')) continue;
        summary = t.replace(/\*+/g, '').replace(/\[(.*?)\]\(.*?\)/g, '$1');
        break;
      }
      out.push({
        slug,
        title,
        type,
        num,
        summary,
        isDeepDive: raw.length > 8000,
      });
    }
  }
  return out.sort((a, b) => {
    if (a.type !== b.type) return a.type === 'classification' ? -1 : 1;
    return a.num.localeCompare(b.num);
  });
}

export function readScenario(type: ScenarioType, slug: string): string | null {
  return safeRead(path.join('scenarios', type, `${slug}.md`));
}

export function readScenarioCatalog(type: ScenarioType): string | null {
  return safeRead(path.join('scenarios', type, 'CATALOG.md'));
}

export function readScenarioMethodology(type: ScenarioType): string | null {
  return safeRead(path.join('scenarios', type, '_methodology.md'));
}

export function getScenarioBySlug(slug: string): Scenario | undefined {
  return listExpertScenarios().find((s) => s.slug === slug);
}

/* ----------------------------------------------------------------------- */
/* Learn — surfaces the comprehensive beginner guide                       */
/* ----------------------------------------------------------------------- */

export interface LearnArticle {
  /** filename without extension, e.g. "01_start_here" — also the URL slug */
  slug: string;
  /** First # heading from the markdown file, falls back to a humanised slug */
  title: string;
  /** One-line summary from index.json */
  summary: string;
  /** Track this article belongs to (foundations / toolkit / methodology) */
  trackId: string;
  /** Track display title */
  trackTitle: string;
  /** True if the markdown file exists on disk; false → "Coming soon" */
  exists: boolean;
}

export interface LearnTrack {
  id: string;
  title: string;
  subtitle: string;
  articles: LearnArticle[];
}

interface LearnIndex {
  tracks: { id: string; title: string; subtitle: string; articles: string[] }[];
  summaries: Record<string, string>;
}

function readLearnIndex(): LearnIndex {
  const file = path.join(CONTENT_ROOT, 'learn', 'index.json');
  return JSON.parse(fs.readFileSync(file, 'utf8')) as LearnIndex;
}

function humaniseSlug(slug: string): string {
  return slug
    .replace(/^\d+_/, '')
    .replace(/_/g, ' ')
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

function extractTitle(markdown: string, fallback: string): string {
  const lines = markdown.split('\n');
  for (const l of lines) {
    if (l.startsWith('# ')) return l.replace(/^#\s+/, '').trim();
  }
  return fallback;
}

export function listLearnTracks(): LearnTrack[] {
  const idx = readLearnIndex();
  return idx.tracks.map((t) => ({
    id: t.id,
    title: t.title,
    subtitle: t.subtitle,
    articles: t.articles.map((slug) => {
      const file = path.join(CONTENT_ROOT, 'learn', `${slug}.md`);
      let exists = false;
      let title = humaniseSlug(slug);
      try {
        const md = fs.readFileSync(file, 'utf8');
        exists = true;
        title = extractTitle(md, title);
      } catch {
        // File doesn't exist yet — coming-soon stub
      }
      return {
        slug,
        title,
        summary: idx.summaries[slug] ?? '',
        trackId: t.id,
        trackTitle: t.title,
        exists,
      };
    }),
  }));
}

export function listLearnArticles(): LearnArticle[] {
  return listLearnTracks().flatMap((t) => t.articles);
}

export function readLearnArticle(slug: string): string | null {
  return safeRead(path.join('learn', `${slug}.md`));
}

export function getLearnArticle(slug: string): LearnArticle | undefined {
  return listLearnArticles().find((a) => a.slug === slug);
}

export function getLearnNeighbours(
  slug: string,
): { prev: LearnArticle | null; next: LearnArticle | null } {
  const all = listLearnArticles().filter((a) => a.exists);
  const idx = all.findIndex((a) => a.slug === slug);
  if (idx === -1) return { prev: null, next: null };
  return {
    prev: idx > 0 ? all[idx - 1] : null,
    next: idx < all.length - 1 ? all[idx + 1] : null,
  };
}

/**
 * Pull a code snippet for a function/class definition from a Python source file.
 * Best-effort: returns the def block by indentation.
 */
export function extractPythonBlock(source: string, name: string): string | null {
  const lines = source.split('\n');
  const headerRegex = new RegExp(`^(\\s*)(def|class)\\s+${name}\\b`);
  let start = -1;
  let baseIndent = 0;
  for (let i = 0; i < lines.length; i++) {
    const m = lines[i].match(headerRegex);
    if (m) {
      start = i;
      baseIndent = m[1].length;
      break;
    }
  }
  if (start === -1) return null;

  let end = lines.length;
  for (let i = start + 1; i < lines.length; i++) {
    const line = lines[i];
    if (line.trim() === '') continue;
    const indent = line.match(/^(\s*)/)?.[1].length ?? 0;
    if (indent <= baseIndent) {
      end = i;
      break;
    }
  }
  return lines.slice(start, end).join('\n');
}
