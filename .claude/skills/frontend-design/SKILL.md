---
name: frontend-design
description: Use this skill when adding new pages, components, or visual elements to the Classical ML From Scratch frontend. It captures the current design system — palette, typography, spacing, card patterns, hover states, motion — so new work stays consistent with what's already shipped. Invoke when the user asks to "add a page", "design a new component", "enhance the look", "apply the theme", or anything that touches visual design in the /frontend directory.
---

# Frontend Design System — Classical ML From Scratch

This document is the source of truth for visual design decisions on the site. Anything new should follow it. Anything inconsistent with it should be flagged and fixed.

---

## 1 — Aesthetic in one paragraph

Minimalist, technical, intentional. Black-and-white in light mode; **neutral charcoal** (Linear / Notion / Vercel style) in dark mode. A single accent colour — **emerald** — used identically in both modes for hover states, focus rings, CTAs, and subtle highlights. Typography is **IBM Plex Sans** for everything plus **IBM Plex Mono** for small uppercase tracking labels and code. No decorative grids, no patterns, no gradients. Visual interest comes from precise spacing, careful borders, and motion on interaction (lift on hover, smooth transitions).

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

These are set via CSS variable overrides inside `.dark { … }` in `globals.css`. Every existing `dark:bg-black`, `dark:bg-gray-900`, `dark:border-gray-800`, etc. resolves to this palette automatically — no per-component edits.

| Role | Value | Tailwind utility that resolves to it |
|---|---|---|
| Body background | `#18181b` (zinc-900) | `dark:bg-black` |
| Card background | `#27272a` (zinc-800) | `dark:bg-gray-900` |
| Strong border | `#3f3f46` (zinc-700) | `dark:border-gray-800` |
| Soft border / hover | `#52525b` (zinc-600) | `dark:border-gray-700` |
| Body text | `#d4d4d8` (zinc-300) | `dark:text-gray-300` |
| Muted text | `#a1a1aa` (zinc-400) | `dark:text-gray-400` |
| Heading text | `#e4e4e7` (zinc-200) | `dark:text-white` |

### Accent — emerald (both modes)

| Use | Light | Dark |
|---|---|---|
| Hover background | `bg-emerald-50` | `dark:bg-emerald-950/40` (or `/20` for cards) |
| Hover border | `hover:border-emerald-600` | `dark:hover:border-emerald-400` |
| Hover inline text | `hover:text-emerald-700` (deep tone) | `dark:hover:text-emerald-300` |
| Focus ring | `focus-visible:outline-emerald-500` | same |
| Strong CTA flash | `bg-emerald-600 text-white` | `bg-emerald-500 text-black` |

**Do not introduce a second chromatic accent** (no blue, no indigo, no amber as primary). Amber is allowed only as a tiny warning/badge accent (e.g., the "Coming soon" tag on Learn cards).

---

## 3 — Typography

Loaded via `next/font/google` in `layout.tsx`. Plex Sans is the default body font; Plex Mono is exposed through the `font-mono` Tailwind utility.

| Where | Font | Tailwind | Notes |
|---|---|---|---|
| Body | Plex Sans (400) | default | Body text uses default leading. |
| Headings | Plex Sans (700) | `font-orbitron` | Class name is historical; it now renders Plex Sans Bold with `letter-spacing: -0.01em`. Keep the class name — many components use it. |
| Page hero H1 / H2 | Plex Sans (700) | `font-orbitron` + `tracking-tight` or `tracking-wider` | Use `tracking-wider` for ALL-CAPS hero titles. |
| Small uppercase labels | Plex Mono (400) | `font-mono` + `text-[10px] tracking-[0.3em] uppercase text-gray-400` | The signature label style of the site. |
| Code / pre | Plex Mono (400) | `font-mono` | Inline code uses `code:not(pre code)` styling from `globals.css`. |

When you need an uppercase tracked label (very common), this is the canonical class:
```html
<div className="text-[10px] font-mono tracking-[0.3em] uppercase text-gray-400">
  PHASE 01
</div>
```

---

## 4 — Spacing rhythm

| Context | Padding | Notes |
|---|---|---|
| Page hero section | `py-12 md:py-16` | Almost always paired with `border-b border-gray-200 dark:border-gray-800`. |
| Page content sections | `py-10` or `py-12` | Section spacing — generous. |
| Inner content widths | `max-w-4xl mx-auto px-4 md:px-6` (text) or `max-w-7xl mx-auto px-4 md:px-6` (grids) | Articles use 4xl, grids use 7xl. |
| Card padding | `px-5 py-5` (compact) or `p-8 md:p-10` (hero card) | Compact cards in grids; hero cards on the home page. |
| Card grid gap | `gap-3` (compact) or `gap-4`–`gap-6` (medium) or `gap-6 md:gap-8` (hero) | Tighter for many cards, larger for fewer. |

---

## 5 — Cards (the dominant component pattern)

There are three card variants on the site. Use the one that fits.

### A. Compact catalog card (Phase cards, Algorithm grid, Scenario grid)

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
  {/* Decorative corner brackets — top-left + bottom-right */}
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
  Start with Phase 1 — Understand the Problem
  <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
</Link>
```

---

## 6 — Motion

Restraint over flash. The motion vocabulary is small:

- **Hover lift**: `hover:-translate-y-0.5` (compact) or `hover:-translate-y-1` (hero), paired with `hover:shadow-md` / `hover:shadow-xl`.
- **Arrow shift**: `group-hover:translate-x-1 transition-transform` for `→` arrows; `group-hover:translate-x-0.5 group-hover:-translate-y-0.5` for `↗` (ArrowUpRight) corner icons.
- **Color transitions**: `transition-colors duration-200` everywhere a hover state changes color.
- **Combined**: `transition-all duration-200` when both transform and color change.

Framer-motion is loaded for the header logo entry only. **Do not add framer-motion to new components** unless the user explicitly asks — keep the bundle lean.

---

## 7 — Iconography

[Lucide React](https://lucide.dev). Sizes mostly `w-3.5 h-3.5` (small inline), `w-4 h-4` (default), `w-5 h-5` (track headers), `w-6 h-6` (hero card markers).

Stock vocabulary already used:
- `BookOpen` — Learn
- `Sparkles` — Practice / AI / generative
- `Layers` — Pipeline phases
- `ArrowRight` — primary forward CTA
- `ArrowUpRight` — card corner affordance
- `ArrowLeft` — back navigation
- `Cpu` — site logo (do not change)
- `Loader2` — spinner
- `RefreshCcw` — reset
- `FlaskConical` — sample / experiment
- `Send` — submit

Pick from these before introducing new icons.

---

## 8 — Don't list

Things that have been intentionally removed — do not add them back.

- ❌ **HeroGrid** — the dotted/lined background overlay. Deleted from the codebase. Hero sections render on the plain page background.
- ❌ **Blue / navy dark theme** — replaced by neutral charcoal. Do not reintroduce blue as a primary or accent color.
- ❌ **Background grids / patterns / textures** of any kind in hero sections.
- ❌ **Orbitron font** — the class name `.font-orbitron` is kept for backwards-compat but it renders Plex Sans Bold. Do not actually import Orbitron again.
- ❌ **Anthropic SDK** / API routes — the site is fully static for GitHub Pages. Do not add new API routes.
- ❌ **Background-gradient text** (`bg-clip-text text-transparent bg-gradient-...`) — keeps the design minimal.
- ❌ **Glassmorphism / backdrop-blur on cards** — the header already uses `backdrop-blur-md`; cards stay solid.
- ❌ **GitHub / Scenarios / Pipelines** in the top nav. Routes still work; nav exposes Learn, Practice, Algorithms only.

---

## 9 — Accessibility minimums

- **Focus**: emerald focus rings on every interactive element. Use `focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-emerald-500` (or rely on the global rule in `globals.css` if present).
- **Contrast**: every text colour above passes WCAG AA on its expected background. Don't introduce new color pairings without checking.
- **Touch targets**: hover-only interactions still need a visible focus state. Lift effects must work without hover (i.e., they should appear on `:focus-visible` too if the element is interactive).

---

## 10 — Mental model for new work

When asked to add or enhance:

1. **First question**: does an existing component or pattern fit? If so, copy it.
2. **Second question**: does the change respect the colour palette, typography, and spacing rhythm above? If not, adjust the change, not the system.
3. **Third question**: is the addition meaningful, or is it polish for polish's sake? The site is minimalist on purpose. Cluttering it with motion, decoration, or gratuitous components erodes the brand.
4. **Last check**: would this still look right with `dark:` removed (light mode) and with `dark:` added (dark mode)? Cards have to work in both. Test mentally.

---

## 11 — Quick reference: Tailwind class snippets to copy

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
| Section content wrap | `max-w-7xl mx-auto px-4 md:px-6 py-10` (grid) or `max-w-4xl mx-auto px-4 md:px-6 py-10` (article) |
| Hero section | `py-12 md:py-16 border-b border-gray-200 dark:border-gray-800` |

---

## 12 — When in doubt

Read the existing implementation. The home page (`src/app/page.tsx`), the Learn page (`src/app/learn/page.tsx`), and the article template (`src/app/learn/[slug]/page.tsx`) together demonstrate every pattern. Match what's there.
