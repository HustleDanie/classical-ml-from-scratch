import type { NextConfig } from "next";
import path from "node:path";

/**
 * Static-export build for GitHub Pages.
 *
 * basePath / assetPrefix are read from env so local `npm run dev` works
 * without paths. CI sets `NEXT_PUBLIC_BASE_PATH=/<repo-name>` for the
 * project-site URL `<user>.github.io/<repo>`.
 *
 * For a user-site (`<user>.github.io`) or a custom domain, leave the env
 * var unset and the basePath becomes empty.
 */
const basePath = process.env.NEXT_PUBLIC_BASE_PATH ?? "";

const nextConfig: NextConfig = {
  output: "export",
  basePath: basePath || undefined,
  assetPrefix: basePath ? `${basePath}/` : undefined,
  trailingSlash: true,
  images: {
    unoptimized: true,
  },
  // KaTeX-heavy scenario pages exceed the default 60s on slow local machines.
  // CI (Linux) finishes well under this limit; bumping for local parity.
  staticPageGenerationTimeout: 180,
  turbopack: {
    root: path.resolve(__dirname),
  },
};

export default nextConfig;
