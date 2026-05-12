# Frontend Design System — Classical ML From Scratch

This is the project-specific design reference for this site. The **generic** `frontend-design` skill (at `.claude/skills/frontend-design/`) is the Anthropic-published skill for any new frontend creative work; this doc captures the decisions already shipped on this particular project so new work stays consistent with what exists.

---

## 1 — Aesthetic in one paragraph

Minimalist, technical, intentional. Black-and-white in light mode; **neutral charcoal** (Linear / Notion / Vercel-style — Tailwind zinc) in dark mode. A single brand accent — **emerald** — used identically in both modes for hover states, focus rings, CTAs, and subtle highlights. Typography is **IBM Plex Sans** for everything plus **IBM Plex Mono** for small uppercase tracking labels and code. No decorative grids, no patterns, no gradients. Visual interest comes from precise spacing, careful borders, and motion on interaction (lift on hover, smooth transitions).

---

## 2 — Colour palette

### Light mode

| Role | Value | Tailwind |
|---|---|---|
| Body background | `#ffffff` | `bg-white` |
| Card background | `#ffffff` | `bg-white` |
| Strong border / accent | `#000000` | `border-black` / `bg-black` |
| Soft border | `#e5e7eb` | `border-gray-200` |
| Medium border | `#d1d5db` | `border-gray-300` |
| Heading text | `#000000` | `text-black` |
| Body text | `#1f2937` | (default `text-foreground`) |
| Muted text | `#6b7280` | `text-gray-500` |
| Caption / mono labels | `#9ca3af` | `text-gray-400` |

### Dark mode — neutral charcoal (Tailwind zinc scale)

Set via CSS variable overrides inside `.dark { … }` in `globals.css`. Every existing `dark:bg-black`, `dark:bg-gray-900`, `dark:border-gray-800`, etc. resolves to this palette automatically.

| Role | Value | Tailwind utility |
|---|---|---|
| Body background | `#18181b` (zinc-900) | `dark:bg-black` |
| Card background | `#27272a` (zinc-800) | `dark:bg-gray-900` |
| Strong border | `#3f3f46` (zinc-700) | `dark:border-gray-800` |
| Soft border / hover | `#52525b` (zinc-600) | `dark:border-gray-700` |
| Body text | `#d4d4d8` (zinc-300) | `dark:text-gray-300` |
| Muted text | `#a1a1aa` (zinc-400) | `dark:text-gray-400` |
| Heading text | `#e4e4e7` (zinc-200) | `dark:text-white` |

### Brand accent — emerald (both modes)

| Use | Light | Dark |
|---|---|---|
| Hover background | `bg-emerald-50` | `dark:bg-emerald-950/40` (`/20` for compact cards) |
| Hover border | `hover:border-emerald-600` | `dark:hover:border-emerald-400` |
| Hover inline text | `hover:text-emerald-700` (deep tone) | `dark:hover:text-emerald-300` |
| Focus ring | `outline-emerald-500` | `outline-emerald-400` |
| Strong CTA flash | `bg-emerald-600 text-white` | `bg-emerald-500 text-black` |

### Semantic-encoding accents (used sparingly)

- **Amber** — used for badges (e.g. "Coming soon" tag on Learn cards) and for the regression semantic accent on `/scenarios` (paired with emerald for classification).
- **The 6 `CATEGORY_ACCENT` colours in `src/lib/algorithms.ts`** — Classification/Regression/Both/Ensemble/Clustering/DimReduction — small functional badges on algorithm cards. Treat as labels, not brand decoration.

Do not introduce additional chromatic accents beyond these.

---

## 3 — Typography

Loaded via `next/font/google` in `layout.tsx`.

| Where | Font | Tailwind |
|---|---|---|
| Body | Plex Sans (400) | default |
| Headings | Plex Sans (700) | `font-orbitron` (class name is historical — renders Plex Sans Bold + `letter-spacing: -0.01em`) |
| Page hero H1 / H2 | Plex Sans (700) | `font-orbitron` + `tracking-tight` or `tracking-wider` |
| Small uppercase labels | Plex Mono (400) | `font-mono` + `text-[10px] tracking-[0.3em] uppercase text-gray-400` |
| Code / pre | Plex Mono (400) | `font-mono` |

The canonical uppercase mono label class:

```html
<div className="text-[10px] font-mono tracking-[0.3em] uppercase text-gray-400">
  PHASE 01
</div>
```

---

## 4 — Spacing rhythm

| Context | Padding |
|---|---|
| Page hero section | `py-12 md:py-16` + `border-b border-gray-200 dark:border-gray-800` |
| Page content sections | `py-10` or `py-12` |
| Inner content widths | `max-w-4xl mx-auto px-4 md:px-6` (text) / `max-w-7xl mx-auto px-4 md:px-6` (grids) |
| Card padding | `px-5 py-5` (compact) / `p-8 md:p-10` (hero card) |
| Card grid gap | `gap-3` (many) · `gap-4` (medium) · `gap-6 md:gap-8` (hero) |

---

## 5 — Card variants

### A. Compact catalog card (phase cards, scenarios, algorithms)

```tsx
<Link
  href={...}
  className="group flex h-full flex-col border border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-900 px-5 py-5 hover:border-emerald-600 dark:hover:border-emerald-400 hover:bg-emerald-50/50 dark:hover:bg-emerald-950/20 hover:-translate-y-0.5 hover:shadow-md dark:hover:shadow-emerald-950/40 transition-all duration-200"
>
  <div className="flex items-center justify-between mb-3">
    <div className="text-[10px] font-mono tracking-[0.3em] uppercase text-gray-400 group-hover:text-emerald-700 dark:group-hover:text-emerald-300 transition-colors">
      {label}
    </div>
    <ArrowUpRight className="w-4 h-4 text-gray-300 dark:text-gray-700 group-hover:text-emerald-600 dark:group-hover:text-emerald-400 group-hover:translate-x-0.5 group-hover:-translate-y-0.5 transition-all" />
  </div>
  <h3 className="text-base font-semibold leading-snug mb-2 group-hover:text-emerald-800 dark:group-hover:text-emerald-200 transition-colors">
    {title}
  </h3>
  {summary && (
    <p className="text-xs text-gray-500 dark:text-gray-400 leading-relaxed line-clamp-3 flex-1">
      {summary}
    </p>
  )}
</Link>
```

### B. Hero card (Home page LEARN / PRACTICE)

```tsx
<Link
  href={...}
  className="group relative block border-2 border-black dark:border-white bg-white dark:bg-gray-900 text-black dark:text-white p-8 md:p-10 hover:bg-emerald-50 dark:hover:bg-emerald-950/40 hover:border-emerald-600 dark:hover:border-emerald-400 hover:-translate-y-1 hover:shadow-xl dark:hover:shadow-emerald-950/50 transition-all duration-200"
>
  <div className="absolute -top-1 -left-1 w-3 h-3 border-t-2 border-l-2 border-black dark:border-white group-hover:border-emerald-600 dark:group-hover:border-emerald-400" />
  <div className="absolute -bottom-1 -right-1 w-3 h-3 border-b-2 border-r-2 border-black dark:border-white group-hover:border-emerald-600 dark:group-hover:border-emerald-400" />
  {/* … label, title, description, CTA … */}
</Link>
```

### C. CTA button (centered link styled like a hero card border)

```tsx
<Link
  href={...}
  className="group inline-flex items-center gap-3 px-6 py-3 border-2 border-black dark:border-white bg-white dark:bg-gray-900 text-black dark:text-white text-xs uppercase tracking-widest font-semibold hover:bg-emerald-50 dark:hover:bg-emerald-950/40 hover:border-emerald-600 dark:hover:border-emerald-400 hover:text-emerald-800 dark:hover:text-emerald-200 transition-colors"
>
  Start with Phase 1
  <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
</Link>
```

---

## 6 — Motion vocabulary

Restraint over flash. Small vocabulary:

- **Hover lift**: `hover:-translate-y-0.5` (compact) or `hover:-translate-y-1` (hero), with `hover:shadow-md` / `hover:shadow-xl`.
- **Arrow shift**: `group-hover:translate-x-1 transition-transform` (`→`) or `group-hover:translate-x-0.5 group-hover:-translate-y-0.5` (`↗`).
- **Colour transitions**: `transition-colors duration-200` everywhere a hover changes colour.
- **Combined**: `transition-all duration-200` when both transform and colour change.

Framer-motion is used for the header logo entrance and for staggered catalog-card entrances (capped at 8 staggered to avoid drag on long lists). **Don't add framer-motion to new components** unless the user explicitly asks.

---

## 7 — Iconography

[Lucide React](https://lucide.dev). Established vocabulary:

- `BookOpen` — Learn
- `Sparkles` — Practice
- `Layers` — Pipeline phases
- `Workflow` — Pipelines
- `ArrowRight` — primary forward CTA
- `ArrowUpRight` — card corner affordance
- `ArrowLeft` — back navigation
- `Cpu` — site logo (do not change)
- `Loader2` — spinner
- `RefreshCcw` — reset
- `FlaskConical` — sample / experiment
- `Send` — submit

Pick from these before introducing new icons. Default sizes: `w-3.5 h-3.5` (small), `w-4 h-4` (default), `w-5 h-5` (track header), `w-6 h-6` (hero card marker).

---

## 8 — Don't list

Things intentionally removed — do not reintroduce.

- ❌ **HeroGrid** — the dotted/lined background overlay. Hero sections render on the plain page background.
- ❌ **Blue / navy dark theme** — replaced by neutral charcoal. Blue is not a brand or accent colour anywhere.
- ❌ **Background grids / patterns / textures / glitch overlays** in hero sections or on card images.
- ❌ **Orbitron font** (`.font-orbitron` class is kept for backwards-compat but renders Plex Sans Bold).
- ❌ **Anthropic SDK / API routes** — the site is fully static for GitHub Pages.
- ❌ **Background-gradient text** (`bg-clip-text text-transparent bg-gradient-…`).
- ❌ **Glassmorphism / backdrop-blur on cards** (the header uses `backdrop-blur-md`; cards stay solid).
- ❌ **GitHub / Scenarios / Pipelines** in the top nav. Routes still work via direct URL.

---

## 9 — Accessibility

- A global `:focus-visible` outline rule lives in `globals.css` (emerald-500 light / emerald-400 dark, 2 px, 2 px offset). Keyboard nav visible everywhere.
- All hover-only interactions must also surface a visible focus state — `:focus-visible` covers it automatically.
- Contrast: every text colour above passes WCAG AA on its expected background.

---

## 10 — Quick reference (Tailwind snippets)

| Need | Class |
|---|---|
| Uppercase mono label | `text-[10px] font-mono tracking-[0.3em] uppercase text-gray-400` |
| Heading H1 (hero) | `font-orbitron text-3xl md:text-5xl font-bold tracking-tight` |
| Heading H2 (section) | `font-orbitron text-xl md:text-2xl font-bold tracking-wider` |
| Body paragraph | `text-gray-600 dark:text-gray-300 text-base leading-relaxed` |
| Card border (compact) | `border border-gray-200 dark:border-gray-800` |
| Card border (hero) | `border-2 border-black dark:border-white` |
| Card background | `bg-white dark:bg-gray-900` |
| Card hover (compact) | `hover:border-emerald-600 dark:hover:border-emerald-400 hover:bg-emerald-50/50 dark:hover:bg-emerald-950/20 hover:-translate-y-0.5 hover:shadow-md transition-all duration-200` |
| Section content wrap | `max-w-7xl mx-auto px-4 md:px-6 py-10` (grid) · `max-w-4xl mx-auto px-4 md:px-6 py-10` (article) |
| Hero section | `py-12 md:py-16 border-b border-gray-200 dark:border-gray-800` |

---

## 11 — When in doubt

Read the existing implementation. The home page (`src/app/page.tsx`), the Learn page (`src/app/learn/page.tsx`), and the article template (`src/app/learn/[slug]/page.tsx`) together demonstrate every pattern. Match what's there.
