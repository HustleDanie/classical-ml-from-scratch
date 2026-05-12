# Classical ML From Scratch — Project Notes for Claude

The site is a beginner-grade walkthrough of the classical-ML pipeline: a 7-phase Learn track, a sample-mode Practice page, a 155-entry expert-scenario catalog, and 3 production pipelines. Fully static, deployed to GitHub Pages.

The live site: **https://hustledanie.github.io/classical-ml-from-scratch/**

Note on the Python library: 12 classical-ML algorithms still live in `/algorithms/<name>/*.py` at the repo root with their plots. They are no longer surfaced on the site — the `/algorithms` route, the per-algorithm READMEs, and all top-level `HOW_IT_WORKS_*.md` / `MODEL_SELECTION_GUIDE_*.md` etc. guide files have been deleted to keep the surface lean. The Python code remains as a code library for anyone reading the repo directly.

---

## Repo layout

```
/algorithms/              # Python implementations + plots (no longer on the site)
/expert_scenarios/        # 155 walkthroughs (classification + regression catalogs)
/real_world_practice/     # 3 production pipelines (Titanic, Ames Housing, NLP)

/frontend/                # Next.js static site
  content/                # Synced from parent repo via scripts/sync-content.mjs (gitignored)
    learn/                # EXCEPTION: hand-written, NOT synced, IS tracked in git
  public/plots/           # Synced PNGs (gitignored)
  scripts/sync-content.mjs
  src/app/                # Next.js App Router pages
  src/components/         # React components
  src/lib/                # Helpers (content.ts, pipelines.ts, etc.)
  DESIGN_SYSTEM.md        # Project-specific design conventions

/.claude/skills/frontend-design/   # Anthropic-published frontend-design skill
/.github/workflows/deploy.yml      # GitHub Pages deploy
```

---

## Key commands

From `/frontend`:

```bash
npm install
npm run dev            # local dev; runs sync-content first
npm run build          # static export (sets output: 'export' via next.config.ts)
npm run sync           # rerun the content sync explicitly
```

For a production build with the correct project-site base path:

```powershell
$env:NEXT_PUBLIC_BASE_PATH = "/classical-ml-from-scratch"; npx next build
```

---

## How content flows

`frontend/scripts/sync-content.mjs` runs on every `npm run build` / `npm run dev` (via `prebuild` / `predev`). It:

1. **Nukes** the synced subdirs of `frontend/content/` (`scenarios`, `pipelines`) and `frontend/public/plots/`.
2. **Preserves** `frontend/content/learn/` — hand-written, tracked in git.
3. **Repopulates** from the parent repo:
   - `/expert_scenarios/<type>/catalog/*.md` → `frontend/content/scenarios/<type>/`
   - `/real_world_practice/<name>/` → `frontend/content/pipelines/<name>/` + plots

The `.gitignore` uses `content/*` (not `content/`) so the `!content/learn/**` re-include rules actually take effect — git won't descend into a fully-ignored dir.

---

## Frontend tech stack

- **Next.js 16** with Turbopack and App Router
- **React 19**, **TypeScript 5**
- **Tailwind 4** (CSS-variable based; `@theme` in globals.css)
- **IBM Plex Sans + IBM Plex Mono** via `next/font/google`
- **react-markdown + remark-gfm + remark-math + rehype-katex** for markdown rendering
- **react-syntax-highlighter** for code blocks
- **framer-motion** for the header animation (used sparingly)
- **next-themes** for dark mode (class strategy)

No Anthropic SDK or any API runtime — the site is **fully static**. GitHub Pages serves it.

---

## Design system

Two relevant docs:

- **`frontend/DESIGN_SYSTEM.md`** — project-specific reference: this site's exact palette, typography, card patterns, spacing rhythm, motion vocabulary, and "don't list". Read this before adding components or changing visual design.
- **`.claude/skills/frontend-design/SKILL.md`** — the Anthropic-published `frontend-design` skill (generic creative-direction guidance for any new frontend work). Invoke when starting a from-scratch design; constrain its output with the project doc above.

Highlights from `DESIGN_SYSTEM.md`:

- **Light mode**: white background, gray-200/300 borders, gray-500/600 muted text, black accents.
- **Dark mode**: neutral charcoal (zinc-900 / zinc-800 / zinc-700) — NO blue cast, NO pure black. Linear/Notion/Vercel-style.
- **Accent**: **emerald** (same in both modes — hover backgrounds, borders, focus rings, CTAs).
- **Typography**: IBM Plex Sans for everything; IBM Plex Mono for small uppercase tracking labels and code.
- **Backwards-compat aliases**: the `.font-orbitron` and `.font-space-mono` Tailwind classes were kept and now point at IBM Plex Sans Bold and IBM Plex Mono respectively — don't rename them site-wide.
- **No HeroGrid** — the dotted/lined background overlay has been deleted. Do not re-add a hero pattern; the design is intentionally clean.
- **Cards**: white in light / `gray-900` (zinc-800) in dark, with a 1- or 2-px border. Hover lifts with `hover:-translate-y-0.5 hover:shadow-md` and shifts the border + key inline text to emerald.

---

## Routes

```
/                              Home — two-card landing (Learn + Practice)
/learn                         The pipeline guide hub — 7 phase cards + CTA
/learn/04_phase_1_understand_problem   Phase 1
/learn/05_phase_2_data_exploration_cleaning    ... through 10_phase_7_deployment
/practice                      Offline sample-mode brief + reveal solution
/scenarios                     155 walkthroughs catalog
/scenarios/[type]/[slug]       Per-scenario
/pipelines                     3 production pipelines
/pipelines/[slug]              Per-pipeline
```

The top nav exposes only **Learn / Practice**. The `/algorithms` route was deleted entirely. Scenarios, Pipelines, and the GitHub link were intentionally removed from the nav (the scenarios + pipelines routes still work via direct URL and are linked from the Footer).

---

## Deployment

GitHub Actions (`.github/workflows/deploy.yml`) deploys on every push to `master`, `main`, or `expert-scenarios-library`. It:

1. Reads `actions/configure-pages@v5` with `enablement: true` to bootstrap Pages.
2. Runs `npm ci` + `npm run build` with `NEXT_PUBLIC_BASE_PATH` injected from `configure-pages` output (= `/classical-ml-from-scratch`).
3. Adds `.nojekyll` to `out/`.
4. Uploads `frontend/out` as the Pages artifact, then `actions/deploy-pages@v4`.

The `github-pages` environment had its `deployment_branch_policy` cleared so any branch can deploy.

---

## Common pitfalls

- **Don't preprocess before `train_test_split`** — that's a teaching theme of the site itself; correspondingly, **don't write content for `content/learn/` from inside the sync script**. That folder is hand-curated.
- **`out/` is gitignored.** Build artifacts must not be committed.
- **`.next/` is gitignored** at both `frontend/` and repo root.
- **Static export limitations**: no API routes, no middleware, no on-demand image optimization. The `next/font` system is fine.
- **KaTeX rendering is slow on Windows.** `next.config.ts` sets `staticPageGenerationTimeout: 180` so local builds don't fail on the 60s default while individual scenarios render math. CI doesn't need it.

---

## Editing conventions

- Match the existing card pattern when adding any new card: 1-px or 2-px border + `bg-white dark:bg-gray-900` + emerald hover (`hover:border-emerald-600 dark:hover:border-emerald-400`).
- Keep `dark:` utility usage — the Tailwind colour overrides in `.dark` (in globals.css) make `dark:bg-gray-900` / `dark:border-gray-800` / etc. resolve to the charcoal palette automatically. Do not switch to arbitrary hex values like `dark:bg-[#27272a]`.
- For long-form article pages, mount `<ReadingProgress />` at the top — already wired into `learn/[slug]`, `scenarios/[type]/[slug]`, `algorithms/[slug]`.
- For prev/next nav inside articles, use the `getLearnNeighbours` / similar helpers; for the first article in a track, fall back to the track's index page.

---

## Branch state

Working branch: `expert-scenarios-library` (Pages env allows it to deploy).

`master` exists on `origin` and pre-dates the frontend — most frontend work lives only on `expert-scenarios-library`. Merge when stable.
