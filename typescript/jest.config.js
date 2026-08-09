module.exports = {
  preset: "ts-jest",
  testEnvironment: "node",
  testMatch: ["**/test/**/*.test.ts"],
  moduleFileExtensions: ["ts", "js", "json", "node"],
  moduleNameMapper: {
    "^@/(.*)$": "<rootDir>/src/$1",
  },
  testPathIgnorePatterns: [
    "/node_modules/",
    // Integration suites run under the Ts Integration workflow.
    "/test/test-integrations/",
    // Live Confident AI suites run under the Ts Confident Tests workflow.
    "/test/test-confident/",
  ],
};

