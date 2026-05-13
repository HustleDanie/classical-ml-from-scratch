import { defineConfig, globalIgnores } from "eslint/config";
import nextVitals from "eslint-config-next/core-web-vitals";
import nextTs from "eslint-config-next/typescript";

const eslintConfig = defineConfig([
  ...nextVitals,
  ...nextTs,
  {
    // Vercel sometimes resolves a different eslint-plugin-react-hooks
    // version than local installs. If a referenced rule like
    // `react-hooks/set-state-in-effect` doesn't exist in that version, the
    // matching disable directive becomes "unused" and the build can fail
    // under strict warning handling. Turn the reporter off so this can
    // never break a deploy.
    linterOptions: { reportUnusedDisableDirectives: "off" },
  },
  globalIgnores([
    ".next/**",
    "out/**",
    "build/**",
    "next-env.d.ts",
    "content/**",
    "public/plots/**",
  ]),
]);

export default eslintConfig;
