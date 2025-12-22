# AGENTS
## Commands
- Package manager: pnpm (package.json packageManager).
- Dev: pnpm dev (alias: pnpm start).
- Build: pnpm build (runs astro check && astro build).
- Preview: pnpm preview.
- Typecheck only: pnpm astro check.
- Lint/tests: not configured; no single-test runner available.
## Code Style
- TypeScript strict via astro/tsconfigs/strict; avoid any and keep export types explicit.
- ES modules throughout; prefer named exports for shared utilities.
- Import grouping: external then local with blank lines where present.
- Formatting: match file-local style; TS/Astro mostly omit semicolons and use double quotes.
- Indentation appears 2 spaces in TS/config; keep consistent within file.
- Naming: camelCase vars/functions, PascalCase types/components; keep existing exceptions (e.g., statusError).
- Error handling: throw explicit errors; avoid empty catch blocks.
- Data fetching: check response.ok before JSON parse; keep generic return types.
- Layout: pages in src/pages, components in src/components, styles in src/styles.
- Astro config: astro.config.mjs uses Tailwind via Vite and sitemap/node adapters.
- Cursor/Copilot rules: none found in .cursor/rules, .cursorrules, or .github/copilot-instructions.md.
