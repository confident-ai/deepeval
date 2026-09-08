// Dedicated Jest config for the native @openrouter/sdk tracing tests.
//
// @openrouter/sdk ships ESM only (its package.json resolves "." straight to
// ./esm/index.js, with no CommonJS build), so it can't be require()d under the
// repo's default CommonJS ts-jest setup. Like jest.mastra.config.cjs, this uses
// babel-jest to transpile everything — the tests, deepeval's sources, and that
// ESM-only package — down to CommonJS. The global jest.config.js is untouched.
//
// Note this only affects tests: deepeval's own OpenRouter integration never
// imports the SDK. It patches whatever client instance the caller hands it, so
// the shipped package has no ESM/CJS constraint of its own.
//
// The suite stubs all HTTP, so no API keys are needed.
//
// Run: npx jest -c jest.openrouter.config.cjs

const babelConfig = {
  presets: [
    ["@babel/preset-env", { targets: { node: "current" } }],
    "@babel/preset-typescript",
  ],
};

/** @type {import('jest').Config} */
module.exports = {
  testEnvironment: "node",
  testMatch: ["**/test/test-integrations/test-openrouter/**/*.test.ts"],
  moduleFileExtensions: ["ts", "js", "mjs", "cjs", "json", "node"],
  moduleNameMapper: {
    "^@/(.*)$": "<rootDir>/src/$1",
  },
  transform: {
    "^.+\\.(tsx?|jsx?|mjs|cjs)$": ["babel-jest", babelConfig],
  },
  transformIgnorePatterns: [],
};
