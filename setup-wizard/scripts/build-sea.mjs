import { chmod, copyFile, mkdir } from "node:fs/promises";
import { execFileSync } from "node:child_process";
import { basename, resolve } from "node:path";

const outputFlag = process.argv.indexOf("--output");
const outputName =
  outputFlag >= 0 && process.argv[outputFlag + 1]
    ? basename(process.argv[outputFlag + 1])
    : "deepeval-setup";
const root = resolve(import.meta.dirname, "..");
const outputDirectory = resolve(root, "release");
const output = resolve(outputDirectory, outputName);
const blob = resolve(root, "dist/deepeval-setup.blob");
const postject = resolve(
  root,
  "node_modules/.bin",
  process.platform === "win32" ? "postject.cmd" : "postject",
);

execFileSync("npm", ["run", "build"], { cwd: root, stdio: "inherit" });
await mkdir(outputDirectory, { recursive: true });
execFileSync(
  process.execPath,
  ["--experimental-sea-config", "sea-config.json"],
  {
    cwd: root,
    stdio: "inherit",
  },
);
/**
 * `postject` cannot inject into a universal binary, which is what nodejs.org
 * installs on macOS, so that one is thinned to the host architecture. A runner
 * whose Node is already single-architecture, as GitHub's tool cache ships it,
 * has nothing to thin and `lipo -thin` would reject it.
 */
const machOArchitectures = (binary) =>
  execFileSync("lipo", ["-archs", binary], { encoding: "utf8" })
    .trim()
    .split(/\s+/);

if (
  process.platform === "darwin" &&
  machOArchitectures(process.execPath).length > 1
) {
  execFileSync(
    "lipo",
    [
      process.execPath,
      "-thin",
      process.arch === "x64" ? "x86_64" : process.arch,
      "-output",
      output,
    ],
    { stdio: "inherit" },
  );
} else {
  await copyFile(process.execPath, output);
}
await chmod(output, 0o755);

const args = [
  output,
  "NODE_SEA_BLOB",
  blob,
  "--sentinel-fuse",
  "NODE_SEA_FUSE_fce680ab2cc467b6e072b8b5df1996b2",
];
if (process.platform === "darwin") {
  args.push("--macho-segment-name", "NODE_SEA");
}
execFileSync(postject, args, { cwd: root, stdio: "inherit" });
if (process.platform === "darwin") {
  execFileSync("codesign", ["--sign", "-", "--force", output], {
    stdio: "inherit",
  });
}

console.log(`Built ${output}`);
