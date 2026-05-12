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
  turbopack: {
    root: path.resolve(__dirname),
  },
};

export default nextConfig;
