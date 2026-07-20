import * as fs from "fs";
import * as path from "path";

type PackageJson = {
  exports: Record<
    string,
    {
      import: string;
      require: string;
      types: string;
    }
  >;
  typesVersions: Record<string, Record<string, string[]>>;
};

const packageRoot = path.resolve(__dirname, "../..");
const packageJsonPath = path.join(packageRoot, "package.json");
const packageJson = JSON.parse(
  fs.readFileSync(packageJsonPath, "utf8"),
) as PackageJson;

const rootTypesVersionKeys = Object.keys(packageJson.typesVersions["*"]).filter(
  (subpath) => subpath !== "*",
);

const subpathExportKeys = Object.keys(packageJson.exports).filter(
  (subpath) => subpath !== ".",
);

describe("package exports", () => {
  test("keeps package exports and typesVersions in sync", () => {
    expect(subpathExportKeys.map((subpath) => subpath.slice(2)).sort()).toEqual(
      rootTypesVersionKeys.sort(),
    );
  });

  test.each(subpathExportKeys)(
    "exports %s with matching runtime and type targets",
    (subpath) => {
      const exportEntry = packageJson.exports[subpath];
      const typesVersionPath = subpath.replace(/^\.\//, "");
      const typesVersionsEntry =
        packageJson.typesVersions["*"][typesVersionPath];

      expect(typesVersionsEntry).toEqual([exportEntry.types.slice(2)]);
      for (const field of ["import", "require", "types"] as const) {
        expect(typeof exportEntry[field]).toBe("string");
        expect(exportEntry[field].length).toBeGreaterThan(0);
      }
    },
  );

  test("exports the documented TypeScript entrypoints", () => {
    for (const subpath of [
      "./metrics",
      "./models",
      "./test-case",
      "./evaluate",
    ]) {
      expect(packageJson.exports[subpath]).toBeDefined();
    }
  });

  test("keeps the legacy camelCase testCase subpath available", () => {
    expect(packageJson.exports["./testCase"]).toEqual({
      import: "./dist/test-case/index.js",
      require: "./dist/test-case/index.js",
      types: "./dist/test-case/index.d.ts",
    });
    expect(packageJson.typesVersions["*"].testCase).toEqual([
      "dist/test-case/index.d.ts",
    ]);
  });
});
