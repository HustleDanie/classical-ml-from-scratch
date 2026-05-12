# Classical ML From Scratch — Project Notes for Claude

The site is a beginner-grade walkthrough of the classical-ML pipeline: a 7-phase Learn track, an AI-powered Practice page, an AI-powered Brief Reading page, a 155-entry expert-scenario catalog, and 3 production pipelines.

Hosted on **Vercel** — switched from GitHub Pages so we can run Node.js API routes that hold the Anthropic API key and call Claude on behalf of the client.

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

For local dev with AI endpoints working, copy `.env.example` → `.env.local` and fill in `ANTHROPIC_API_KEY`. Endpoints under `/api/*` will then succeed instead of 503.

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
/learn                         Two cards: Pipeline Phases + Reading a Brief
/learn/brief-reading           AI brief + 7-phase signal extraction (+ static default)
/learn/[phase_slug]            Phase 1–7 deep-dives (04…10)
/practice                      Generate brief → work on paper → reveal solution
/scenarios                     155 walkthroughs catalog
/scenarios/[type]/[slug]       Per-scenario
/pipelines                     3 production pipelines
/pipelines/[slug]              Per-pipeline

/api/practice/brief            POST → streams a new brief
/api/practice/solution         POST → streams the 17-step expert solution + critique
/api/learn/brief-reading       POST → streams a brief + 7-phase signal extraction
```

The top nav exposes only **Learn / Practice**. The `/algorithms` route was deleted entirely. Scenarios, Pipelines, and the GitHub link were intentionally removed from the nav (the scenarios + pipelines routes still work via direct URL and are linked from the Footer).

---

## Deployment

Vercel. The repo is connected to a Vercel project; every push to any tracked branch triggers an auto-deploy.

**One-time setup:**
1. Sign in at https://vercel.com with the same GitHub account.
2. "Add new… → Project" and import `classical-ml-from-scratch`.
3. **Root directory**: `frontend` (the Next app lives in a sub-folder).
4. Framework preset: Next.js (auto-detected).
5. Environment variables → add `ANTHROPIC_API_KEY` = your key from console.anthropic.com.
6. Deploy.

Vercel runs `npm run build` which fires `prebuild` (the content sync) automatically. No `vercel.json` needed.

**Local dev mirrors prod:** `npm run dev` reads `.env.local` for `ANTHROPIC_API_KEY`, so the `/api/*` endpoints work end-to-end without deploying.

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
