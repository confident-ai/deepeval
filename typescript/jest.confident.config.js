// Jest config for Ts Confident Tests (live API suites).
// Default jest.config.js ignores /test/test-confident/ so unit CI stays offline.
module.exports = {
  preset: "ts-jest",
  testEnvironment: "node",
  testMatch: ["**/test/test-confident/**/*.test.ts"],
  testPathIgnorePatterns: ["/node_modules/"],
  moduleFileExtensions: ["ts", "js", "json", "node"],
  moduleNameMapper: {
    "^@/(.*)$": "<rootDir>/src/$1",
  },
  // See jest.config.js: pin CommonJS so ts-jest skips its TS151002 warning.
  transform: {
    "^.+\\.tsx?$": ["ts-jest", { tsconfig: { module: "commonjs" } }],
  },
};
