// Dedicated Jest config for the Mastra integration tests.
//
// Mastra is ESM-first: its dependency tree includes ESM-only packages
// (@sindresorhus/slugify, @sindresorhus/transliterate, tokenx, ...) that can't be
// require()d under the repo's default CommonJS ts-jest setup. This config uses
// babel-jest to transpile everything (the test/app/deepeval sources AND those
// ESM-only leaf packages) down to CommonJS, so the real @mastra/core runtime
// loads and deepeval's own CJS deps keep working. The global jest.config.js is
// untouched, so the other integration suites are unaffected.
//
// Run:                 npx jest -c jest.mastra.config.cjs
// Generate fixtures:   GENERATE_SCHEMAS=true npx jest -c jest.mastra.config.cjs
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
  testMatch: ["**/test/test-integrations/test-mastra/**/*.test.ts"],
  moduleFileExtensions: ["ts", "js", "mjs", "cjs", "json", "node"],
  moduleNameMapper: {
    "^@/(.*)$": "<rootDir>/src/$1",
  },
  transform: {
    "^.+\\.(tsx?|jsx?|mjs|cjs)$": ["babel-jest", babelConfig],
  },
  transformIgnorePatterns: [],
};
