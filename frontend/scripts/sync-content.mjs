// Sync content from the parent classical-ml-from-scratch repo into the frontend.
//
//  - Plot PNGs   →  public/plots/<algorithm-folder>/*.png
//  - Pipeline plots → public/plots/pipelines/<pipeline-folder>/*.png
//  - HOW_IT_WORKS_*.md → content/how_it_works/<slug>.md
//  - Algorithm READMEs + .py → content/algorithms/<folder>/{README.md,<algo>.py}
//  - Pipeline source files + READMEs → content/pipelines/<folder>/...
//  - EXPERT_SCENARIO_*.md → content/scenarios/<slug>.md
//
// Idempotent: nukes destination directories first, then re-copies.

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const FRONTEND_ROOT = path.resolve(__dirname, '..');
const REPO_ROOT = path.resolve(FRONTEND_ROOT, '..');

const PUBLIC_PLOTS = path.join(FRONTEND_ROOT, 'public', 'plots');
const CONTENT = path.join(FRONTEND_ROOT, 'content');

function rmDir(dir) {
  if (fs.existsSync(dir)) {
    fs.rmSync(dir, { recursive: true, force: true });
  }
}

function ensureDir(dir) {
  fs.mkdirSync(dir, { recursive: true });
}

function copyFile(src, dest) {
  ensureDir(path.dirname(dest));
  fs.copyFileSync(src, dest);
}

function listDir(dir) {
  try {
    return fs.readdirSync(dir, { withFileTypes: true });
  } catch {
    return [];
  }
}

function copyDir(src, dest, filter = () => true) {
  if (!fs.existsSync(src)) return 0;
  let count = 0;
  for (const entry of listDir(src)) {
    const s = path.join(src, entry.name);
    const d = path.join(dest, entry.name);
    if (entry.isDirectory()) {
      count += copyDir(s, d, filter);
    } else if (filter(entry.name, s)) {
      copyFile(s, d);
      count += 1;
    }
  }
  return count;
}

console.log('[sync-content] frontend root:', FRONTEND_ROOT);
console.log('[sync-content] repo root:    ', REPO_ROOT);

// Reset destinations. content/learn/ is hand-written and tracked in git —
// preserve it. Everything else under content/ is regenerated below.
rmDir(PUBLIC_PLOTS);
for (const sub of ['how_it_works', 'scenarios', 'pipelines', 'guides']) {
  rmDir(path.join(CONTENT, sub));
}
ensureDir(PUBLIC_PLOTS);
ensureDir(CONTENT);

// 1. Algorithm sync — REMOVED. The /algorithms route was deleted from the
// site (Phase pages are the canonical learning surface now). Algorithm
// plots and Python source still live in `/algorithms/<name>/` at the
// repo root but are no longer copied into the frontend.

// 2. HOW_IT_WORKS markdown — REMOVED for the same reason. Source files
// have been deleted from the repo root.

// 3. Expert scenarios — pull from expert_scenarios/{classification,regression}/catalog/*.md
//    Output to content/scenarios/<type>/<slug>.md.
//    Also copy CATALOG.md indexes and methodology guides.
const scenariosDest = path.join(CONTENT, 'scenarios');
const scenariosRoot = path.join(REPO_ROOT, 'expert_scenarios');
let scenarioCount = 0;
let catalogCount = 0;
let methodologyCount = 0;

// Top-level scenarios README
const scenariosReadme = path.join(scenariosRoot, 'README.md');
if (fs.existsSync(scenariosReadme)) {
  copyFile(scenariosReadme, path.join(scenariosDest, 'README.md'));
}

for (const type of ['classification', 'regression']) {
  const typeRoot = path.join(scenariosRoot, type);
  if (!fs.existsSync(typeRoot)) continue;

  // catalog files (per-scenario walkthroughs + the CATALOG.md index)
  const catalogSrc = path.join(typeRoot, 'catalog');
  for (const f of listDir(catalogSrc)) {
    if (!f.isFile() || !f.name.endsWith('.md')) continue;
    const dest = path.join(scenariosDest, type, f.name);
    copyFile(path.join(catalogSrc, f.name), dest);
    if (f.name === 'CATALOG.md') {
      catalogCount += 1;
    } else {
      scenarioCount += 1;
    }
  }

  // methodology guide
  const methodologySrc = path.join(typeRoot, 'from_brief_to_solution', 'METHODOLOGY.md');
  if (fs.existsSync(methodologySrc)) {
    copyFile(
      methodologySrc,
      path.join(scenariosDest, type, '_methodology.md'),
    );
    methodologyCount += 1;
  }
}

// Legacy: any EXPERT_SCENARIO_*.md still at REPO_ROOT (e.g., the clustering one)
for (const f of listDir(REPO_ROOT)) {
  if (f.isFile() && /^EXPERT_SCENARIO_.*\.md$/i.test(f.name)) {
    const slug = f.name.replace(/\.md$/i, '').toLowerCase();
    copyFile(
      path.join(REPO_ROOT, f.name),
      path.join(scenariosDest, 'legacy', `${slug}.md`),
    );
    scenarioCount += 1;
  }
}

console.log(
  `[sync-content] copied ${scenarioCount} expert scenarios + ${catalogCount} catalogs + ${methodologyCount} methodology guides`,
);

// 4. Real-world pipelines (source + plots)
const pipelinesSrc = path.join(REPO_ROOT, 'real_world_practice');
let pipeFiles = 0;
let pipePlots = 0;
for (const entry of listDir(pipelinesSrc)) {
  if (!entry.isDirectory()) continue;
  const folder = entry.name;
  const src = path.join(pipelinesSrc, folder);

  // plots → public/plots/pipelines/<folder>/*
  const plotsSrc = path.join(src, 'plots');
  const plotsDest = path.join(PUBLIC_PLOTS, 'pipelines', folder);
  pipePlots += copyDir(plotsSrc, plotsDest, (name) => /\.(png|jpe?g|svg)$/i.test(name));

  // source files (.py, .md) → content/pipelines/<folder>/*
  const contentDest = path.join(CONTENT, 'pipelines', folder);
  for (const f of listDir(src)) {
    if (f.isFile() && (f.name.endsWith('.py') || f.name.endsWith('.md'))) {
      copyFile(path.join(src, f.name), path.join(contentDest, f.name));
      pipeFiles += 1;
    }
  }
}
console.log(`[sync-content] copied ${pipePlots} pipeline plots`);
console.log(`[sync-content] copied ${pipeFiles} pipeline source files`);

// 5. Top-level guide markdown — REMOVED. All top-level guide .md files
// have been deleted; the canonical reference is now content/learn/.

console.log('[sync-content] done.');
