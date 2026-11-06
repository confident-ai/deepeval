import { defineConfig } from "vitest/config";

export default defineConfig({
  plugins: [
    {
      name: "markdown-as-text",
      enforce: "pre",
      transform(source, id) {
        if (!id.endsWith(".md")) return undefined;
        return {
          code: `export default ${JSON.stringify(source)};`,
          map: null,
        };
      },
    },
  ],
  test: {
    environment: "node",
    restoreMocks: true,
    /**
     * Assertions read plain text, so color cannot depend on whether the shell
     * or the CI runner happens to enable it. `tests/ui.test.ts` unsets this to
     * assert the palette itself.
     */
    env: { NO_COLOR: "1" },
  },
});
