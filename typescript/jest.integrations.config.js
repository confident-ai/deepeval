// Jest config for Ts Integration (non-Mastra suites).
// Default jest.config.js ignores /test/test-integrations/ so unit CI stays fast.
module.exports = {
  preset: "ts-jest",
  testEnvironment: "node",
  testMatch: ["**/test/test-integrations/**/*.test.ts"],
  testPathIgnorePatterns: [
    "/node_modules/",
    // Mastra needs babel-jest (jest.mastra.config.cjs).
    "/test/test-integrations/test-mastra/",
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
