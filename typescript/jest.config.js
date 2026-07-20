module.exports = {
  testEnvironment: "node",
  testMatch: ["**/test/**/*.test.ts"],
  moduleFileExtensions: ["ts", "js", "json", "node"],
  transform: {
    "^.+\\.tsx?$": ["ts-jest", { isolatedModules: true }],
  },
  moduleNameMapper: {
    "^(\\.\\.?\\/.+)\\.js$": "$1",
  },
  setupFilesAfterEnv: ["<rootDir>/jest.setup.js"],
};
