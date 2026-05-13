import type { NextConfig } from "next";
import path from "node:path";

/**
 * Vercel-hosted Next.js app.
 *
 * Was a static export for GitHub Pages — now back to a full SSR/SSG hybrid
 * so we can ship Node.js API routes that hold the ANTHROPIC_API_KEY and
 * call the Claude API on behalf of the client. Vercel deploys this
 * automatically from the connected GitHub repo.
 */
const nextConfig: NextConfig = {
  // Static-export specific options removed: output, basePath, assetPrefix,
  // trailingSlash. Vercel handles routing and CDN natively.
  images: {
    unoptimized: true,
  },
  // KaTeX-heavy pages exceed the default 60s on slow local machines; CI
  // and Vercel finish well within the limit.
  staticPageGenerationTimeout: 180,
  // The Next app lives in /frontend (a sub-directory of the repo). Vercel
  // would otherwise auto-set outputFileTracingRoot to the parent repo
  // root, which mismatches turbopack.root and fails the build. Pin both
  // explicitly so they agree.
  outputFileTracingRoot: path.resolve(__dirname),
  turbopack: {
    root: path.resolve(__dirname),
  },
};

export default nextConfig;
