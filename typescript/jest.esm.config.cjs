// Jest config for the integration suites whose frameworks ship ESM only.
//
// Mastra's dependency tree includes ESM-only packages (@sindresorhus/slugify,
// @sindresorhus/transliterate, tokenx, ...), and AI SDK 7 ships ESM only. Those
// can't be require()d under the repo's default CommonJS ts-jest setup. This
// config uses babel-jest to transpile everything (the test/app/deepeval sources
// AND those ESM-only packages) down to CommonJS, so the real framework runtimes
// load and deepeval's own CJS deps keep working. The global jest.config.js is
// untouched, so the other integration suites are unaffected.
//
// Run:                 npx jest -c jest.esm.config.cjs test/test-integrations/test-mastra
// Generate fixtures:   GENERATE_SCHEMAS=true npx jest -c jest.esm.config.cjs ...
// Live agent suites require OPENAI_API_KEY; no CONFIDENT_API_KEY needed.

const babelConfig = {
  presets: [
    ["@babel/preset-env", { targets: { node: "current" } }],
    "@babel/preset-typescript",
  ],
};

/** @type {import('jest').Config} */
module.exports = {
  testEnvironment: "node",
  testMatch: [
    "**/test/test-integrations/test-mastra/**/*.test.ts",
    "**/test/test-integrations/test-ai-sdk/**/*.test.ts",
  ],
  moduleFileExtensions: ["ts", "js", "mjs", "cjs", "json", "node"],
  moduleNameMapper: {
    "^@/(.*)$": "<rootDir>/src/$1",
  },
  transform: {
    "^.+\\.(tsx?|jsx?|mjs|cjs)$": ["babel-jest", babelConfig],
  },
  transformIgnorePatterns: [],
};
