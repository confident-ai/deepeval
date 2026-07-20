module.exports = {
  preset: "ts-jest",
  testEnvironment: "node",
  testMatch: ["**/test/**/*.test.ts"],
  moduleFileExtensions: ["ts", "js", "json", "node"],
  testPathIgnorePatterns: [
    "/node_modules/",
    "/test/test-integrations/test-mastra/",
  ],
};
