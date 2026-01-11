import js from "@eslint/js"
import astro from "eslint-plugin-astro"
import oxlint from "eslint-plugin-oxlint"
import tsParser from "@typescript-eslint/parser"
import astroParser from "astro-eslint-parser"

export default [
  {
    ignores: ["dist/", ".astro/", "node_modules/"]
  },
  js.configs.recommended,
  {
    files: ["**/*.{js,mjs,cjs,ts,tsx}"],
    languageOptions: {
      parser: tsParser,
      sourceType: "module"
    }
  },
  ...astro.configs.recommended,
  {
    files: ["**/*.astro"],
    languageOptions: {
      parser: astroParser,
      parserOptions: {
        parser: tsParser,
        extraFileExtensions: [".astro"]
      }
    }
  },
  // Disable overlapping ESLint rules covered by oxlint for speed/consistency.
  ...oxlint.configs["flat/recommended"]
]
