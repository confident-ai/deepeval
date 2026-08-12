import { defineConfig } from "eslint/config";

import eslintConfigPrettier from "eslint-config-prettier/flat";
import js from "@eslint/js";
import globals from "globals";
import tseslint from "typescript-eslint";

export default defineConfig([
  {
    ignores: ["dist/**", "test/**"],
    files: ["**/*.{js,mjs,cjs,ts,mts,cts,tsx}"],
    languageOptions: {
      globals: globals.node,
    },
    extends: [js.configs.recommended, ...tseslint.configs.recommended],
    rules: {
      "@typescript-eslint/no-explicit-any": "off",
      // Best-effort cleanup / optional side effects use empty catch on purpose.
      "no-empty": ["error", { allowEmptyCatch: true }],
      "@typescript-eslint/no-unused-vars": [
        "warn",
        {
          argsIgnorePattern: "^_",
          varsIgnorePattern: "^_",
          caughtErrorsIgnorePattern: "^_",
        },
      ],
    },
  },
  {
    files: ["src/**/*.{ts,mts,cts}"],
    rules: {
      "@typescript-eslint/no-restricted-imports": [
        "error",
        {
          patterns: [
            {
              group: ["./*", "./**", "../*", "../**"],
              message:
                "Use the '@/' alias (rooted at src/) instead of relative imports.",
            },
          ],
        },
      ],
    },
  },
  {
    // Emits ESM with no alias rewriting step, so siblings must be imported
    // relatively with explicit extensions.
    files: ["src/inspect/ui/**/*.{ts,tsx}"],
    rules: {
      "@typescript-eslint/no-restricted-imports": "off",
    },
  },
  {
    files: ["tests/**/*.ts", "examples/vitest/**/*.ts"],
    languageOptions: {
      parserOptions: {
        project: ["./tsconfig.test.json"],
        tsconfigRootDir: import.meta.dirname,
      },
    },
    rules: {
      "@typescript-eslint/no-floating-promises": "error",
    },
  },
  eslintConfigPrettier,
]);
