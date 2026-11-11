// Jest config for Ts Integration (suites whose frameworks load as CommonJS).
// Default jest.config.js ignores /test/test-integrations/ so unit CI stays fast.
module.exports = {
  preset: "ts-jest",
  testEnvironment: "node",
  testMatch: ["**/test/test-integrations/**/*.test.ts"],
  testPathIgnorePatterns: [
    "/node_modules/",
    // ESM-only frameworks need babel-jest (jest.esm.config.cjs).
    "/test/test-integrations/test-mastra/",
    "/test/test-integrations/test-ai-sdk/",
  ],
  moduleFileExtensions: ["ts", "js", "json", "node"],
  moduleNameMapper: {
    "^@/(.*)$": "<rootDir>/src/$1",
  },
  // See jest.config.js: pin CommonJS so ts-jest skips its TS151002 warning.
  transform: {
    "^.+\\.tsx?$": ["ts-jest", { tsconfig: { module: "commonjs" } }],
  },
};
