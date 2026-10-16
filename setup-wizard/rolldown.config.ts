import { defineConfig } from "rolldown";

const shared = {
  input: "src/index.ts",
  platform: "node" as const,
  external: [/^node:/],
  moduleTypes: {
    ".md": "text" as const,
  },
};

export default defineConfig([
  {
    ...shared,
    output: {
      file: "dist/cli.js",
      format: "esm",
    },
  },
  {
    ...shared,
    output: {
      file: "dist/sea.cjs",
      format: "cjs",
    },
  },
]);
