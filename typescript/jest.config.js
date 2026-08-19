module.exports = {
  preset: "ts-jest",
  testEnvironment: "node",
  testMatch: ["**/test/**/*.test.ts"],
  moduleFileExtensions: ["ts", "js", "json", "node"],
  moduleNameMapper: {
    "^@/(.*)$": "<rootDir>/src/$1",
  },
  // Jest runs CommonJS, and this is a CJS package, so "nodenext" already emits
  // CJS here. Saying so explicitly silences ts-jest's TS151002 hybrid-module
  // warning (printed once per test file) without touching the build config.
  transform: {
    "^.+\\.tsx?$": ["ts-jest", { tsconfig: { module: "commonjs" } }],
  },
  testPathIgnorePatterns: [
    "/node_modules/",
    // Integration suites run under the Ts Integration workflow.
    "/test/test-integrations/",
    // Live Confident AI suites run under the Ts Confident Tests workflow.
    "/test/test-confident/",
  ],
};

