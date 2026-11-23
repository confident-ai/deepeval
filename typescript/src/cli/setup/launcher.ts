/**
 * Download, verify, cache and run the `deepeval-setup` wizard binary.
 *
 * The wizard lives in `setup-wizard/` and ships as a standalone executable on
 * GitHub releases tagged `setup-v<version>`. Every run asks GitHub for the
 * newest of those releases, so a wizard fix reaches users without a DeepEval
 * upgrade, and falls back to the newest cached copy when GitHub cannot be
 * reached. Mirrors `deepeval/cli/setup/launcher.py`.
 */
import { createHash } from "crypto";
import { execFile, spawnSync } from "child_process";
import * as fs from "fs";
import * as os from "os";
import * as path from "path";
import { promisify } from "util";
import axios from "axios";

export const BINARY_NAME = "deepeval-setup";
export const CHECKSUMS_NAME = "SHA256SUMS";
export const RELEASES_URL =
  "https://github.com/confident-ai/deepeval/releases/download";
export const RELEASES_API_URL =
  "https://api.github.com/repos/confident-ai/deepeval/releases";
// Python and TypeScript SDK releases share the list, newest first, so the
// newest wizard release is normally on the first page.
export const RELEASES_PER_PAGE = 100;
export const LOOKUP_PAGES = 3;
const TAG = /^setup-v(\d+)\.(\d+)\.(\d+)$/;
const VERSION = /^(\d+)\.(\d+)\.(\d+)$/;

const OS_NAMES: Partial<Record<NodeJS.Platform, string>> = {
  darwin: "darwin",
  linux: "linux",
};
const ARCH_NAMES: Partial<Record<string, string>> = {
  x64: "x64",
  arm64: "arm64",
};
const SHA256 = /^[0-9a-f]{64}$/;

/** A failure to report as a plain message, without a stack trace. */
export class SetupLauncherError extends Error {}

export interface Target {
  os: string;
  arch: string;
}

export function resolveTarget(
  platform: NodeJS.Platform = process.platform,
  arch: string = process.arch,
): Target {
  const osName = OS_NAMES[platform];
  const archName = ARCH_NAMES[arch];
  if (!osName || !archName) {
    throw new SetupLauncherError(
      `deepeval setup supports macOS and Linux on x64 and arm64 for now, not ${platform} on ${arch}.`,
    );
  }
  return { os: osName, arch: archName };
}

export function archiveName(target: Target): string {
  return `${BINARY_NAME}-${target.os}-${target.arch}.tar.gz`;
}

export function releaseUrl(name: string, version: string): string {
  return `${RELEASES_URL}/setup-v${version}/${name}`;
}

/** `${XDG_CACHE_HOME:-~/.cache}/deepeval/setup`. */
export function cacheRoot(
  env: NodeJS.ProcessEnv = process.env,
  home: string = os.homedir(),
): string {
  const configured = env.XDG_CACHE_HOME ?? "";
  // The XDG spec says a relative value is invalid and should be ignored.
  const root = path.isAbsolute(configured)
    ? configured
    : path.join(home, ".cache");
  return path.join(root, "deepeval", "setup");
}

export function cacheDir(
  version: string,
  env: NodeJS.ProcessEnv = process.env,
  home: string = os.homedir(),
): string {
  return path.join(cacheRoot(env, home), version);
}

type VersionParts = [number, number, number];

function parseVersion(pattern: RegExp, text: string): VersionParts | undefined {
  const match = pattern.exec(text);
  return match
    ? [Number(match[1]), Number(match[2]), Number(match[3])]
    : undefined;
}

function compareVersions(a: VersionParts, b: VersionParts): number {
  return a[0] - b[0] || a[1] - b[1] || a[2] - b[2];
}

async function fetchReleases(page: number): Promise<unknown[]> {
  const response = await axios.get<unknown>(RELEASES_API_URL, {
    params: { per_page: RELEASES_PER_PAGE, page },
    headers: { Accept: "application/vnd.github+json" },
    timeout: 30_000,
  });
  if (!Array.isArray(response.data)) {
    throw new SetupLauncherError("GitHub returned an unexpected release list.");
  }
  return response.data;
}

interface ReleaseLike {
  tag_name?: unknown;
  draft?: unknown;
  prerelease?: unknown;
  assets?: unknown;
}

/**
 * The newest published `setup-v*` release that ships `asset`. Drafts and
 * prereleases are skipped, so a release can be staged as a prerelease before
 * it reaches everyone.
 */
export async function latestVersion(
  asset: string,
  fetch: (page: number) => Promise<unknown[]> = fetchReleases,
): Promise<string> {
  let newest: VersionParts | undefined;
  for (let page = 1; page <= LOOKUP_PAGES; page++) {
    const releases = await fetch(page);
    for (const candidate of releases) {
      if (!candidate || typeof candidate !== "object") continue;
      const release = candidate as ReleaseLike;
      if (release.draft || release.prerelease) continue;
      const version = parseVersion(TAG, String(release.tag_name ?? ""));
      if (!version) continue;
      const names = new Set(
        (Array.isArray(release.assets) ? release.assets : []).map(
          (item: unknown) =>
            item && typeof item === "object"
              ? (item as { name?: unknown }).name
              : undefined,
        ),
      );
      if (!names.has(asset) || !names.has(CHECKSUMS_NAME)) continue;
      if (!newest || compareVersions(version, newest) > 0) newest = version;
    }
    if (newest || releases.length < RELEASES_PER_PAGE) break;
  }
  if (!newest) {
    throw new SetupLauncherError(
      `No published setup wizard release ships ${asset}.`,
    );
  }
  return newest.join(".");
}

/** The newest wizard already in the cache, for when GitHub is out of reach. */
export function newestCached(
  env: NodeJS.ProcessEnv = process.env,
  home: string = os.homedir(),
): { version: string; binary: string } | undefined {
  const root = cacheRoot(env, home);
  let entries: string[];
  try {
    entries = fs.readdirSync(root);
  } catch {
    return undefined;
  }
  let newest: { parts: VersionParts; version: string } | undefined;
  for (const entry of entries) {
    const parts = parseVersion(VERSION, entry);
    if (!parts || !isExecutableFile(path.join(root, entry, BINARY_NAME))) {
      continue;
    }
    if (!newest || compareVersions(parts, newest.parts) > 0) {
      newest = { parts, version: entry };
    }
  }
  return newest
    ? {
        version: newest.version,
        binary: path.join(root, newest.version, BINARY_NAME),
      }
    : undefined;
}

/** Read `name`'s digest from `sha256sum` output. */
export function expectedSha256(checksums: string, name: string): string {
  for (const line of checksums.split(/\r?\n/)) {
    const parts = line.trim().split(/\s+/);
    if (parts.length === 2 && parts[1].replace(/^\*/, "") === name) {
      const digest = parts[0].toLowerCase();
      if (SHA256.test(digest)) return digest;
    }
  }
  throw new SetupLauncherError(
    `${CHECKSUMS_NAME} has no valid entry for ${name}.`,
  );
}

async function download(url: string): Promise<Buffer> {
  try {
    const response = await axios.get<ArrayBuffer>(url, {
      responseType: "arraybuffer",
      // An idle timeout: a slow but steady download is not cut off.
      timeout: 60_000,
    });
    return Buffer.from(response.data);
  } catch (error) {
    const reason = axios.isAxiosError(error)
      ? error.response
        ? `${error.response.status} for ${url}`
        : error.message
      : String(error);
    throw new SetupLauncherError(
      `Could not download the setup wizard: ${reason}`,
    );
  }
}

const execFileAsync = promisify(execFile);

/** The system `tar`, as the curl installer already requires, so the SDK
 * needs no archive dependency. Only the binary's own member is extracted. */
async function extractWithTar(
  archivePath: string,
  directory: string,
): Promise<void> {
  try {
    await execFileAsync("tar", [
      "-xzf",
      archivePath,
      "-C",
      directory,
      BINARY_NAME,
    ]);
  } catch (error) {
    throw new SetupLauncherError(
      `Could not extract ${BINARY_NAME} from the setup archive: ${
        error instanceof Error ? error.message : String(error)
      }`,
    );
  }
}

export interface RunResult {
  status: number | null;
  signal: NodeJS.Signals | null;
  error?: Error;
}

function runInherited(binary: string, args: string[]): RunResult {
  return spawnSync(binary, args, { stdio: "inherit" });
}

export interface LauncherDependencies {
  fetchReleases: (page: number) => Promise<unknown[]>;
  download: (url: string) => Promise<Buffer>;
  extract: (archivePath: string, directory: string) => Promise<void>;
  run: (binary: string, args: string[]) => RunResult;
  platform: NodeJS.Platform;
  arch: string;
  env: NodeJS.ProcessEnv;
  home: string;
  notify: (message: string) => void;
}

export const defaultLauncherDependencies = (): LauncherDependencies => ({
  fetchReleases,
  download,
  extract: extractWithTar,
  run: runInherited,
  platform: process.platform,
  arch: process.arch,
  env: process.env,
  home: os.homedir(),
  notify: (message) => process.stderr.write(`${message}\n`),
});

function isExecutableFile(file: string): boolean {
  try {
    if (!fs.statSync(file).isFile()) return false;
    fs.accessSync(file, fs.constants.X_OK);
    return true;
  } catch {
    return false;
  }
}

/**
 * Verify `archive` against `checksums`, then extract the binary to `target`.
 * Nothing is written unless the digest matches, and the binary appears at
 * `target` only by an atomic rename from a directory beside it.
 */
export async function installBinary(
  archive: Buffer,
  checksums: string,
  name: string,
  target: string,
  extract: LauncherDependencies["extract"] = extractWithTar,
): Promise<string> {
  const actual = createHash("sha256").update(archive).digest("hex");
  if (actual !== expectedSha256(checksums, name)) {
    throw new SetupLauncherError(
      `Checksum mismatch for ${name}; refusing to run it. Try again, and report it if it persists.`,
    );
  }

  const directory = path.dirname(target);
  fs.mkdirSync(directory, { recursive: true });
  const staging = fs.mkdtempSync(path.join(directory, `.${BINARY_NAME}.`));
  try {
    const archivePath = path.join(staging, name);
    fs.writeFileSync(archivePath, archive, { mode: 0o600 });
    const extractDirectory = path.join(staging, "extract");
    fs.mkdirSync(extractDirectory);
    await extract(archivePath, extractDirectory);

    const extracted = path.join(extractDirectory, BINARY_NAME);
    let stats: fs.Stats;
    try {
      stats = fs.lstatSync(extracted);
    } catch {
      throw new SetupLauncherError(
        `The setup archive does not contain ${BINARY_NAME}.`,
      );
    }
    if (!stats.isFile()) {
      throw new SetupLauncherError(
        `${BINARY_NAME} in the setup archive is not a regular file.`,
      );
    }
    fs.chmodSync(extracted, 0o755);
    fs.renameSync(extracted, target);
  } finally {
    fs.rmSync(staging, { recursive: true, force: true });
  }
  return target;
}

function describeError(error: unknown): string {
  if (axios.isAxiosError(error)) {
    return error.response
      ? `GitHub answered ${error.response.status}`
      : error.message;
  }
  return error instanceof Error ? error.message : String(error);
}

/** Return the newest wizard binary, downloading it first if needed. */
export async function ensureBinary(
  dependencies: LauncherDependencies,
): Promise<string> {
  const target = resolveTarget(dependencies.platform, dependencies.arch);
  const name = archiveName(target);

  let version: string;
  try {
    version = await latestVersion(name, dependencies.fetchReleases);
  } catch (error) {
    const cached = newestCached(dependencies.env, dependencies.home);
    if (!cached) {
      throw new SetupLauncherError(
        `Could not find the setup wizard on GitHub: ${describeError(error)}`,
      );
    }
    dependencies.notify(
      `Could not check GitHub for a newer setup wizard (${describeError(error)}); using the cached ${cached.version}.`,
    );
    return cached.binary;
  }

  const binary = path.join(
    cacheDir(version, dependencies.env, dependencies.home),
    BINARY_NAME,
  );
  if (isExecutableFile(binary)) return binary;

  dependencies.notify(
    `Downloading the DeepEval setup wizard ${version} (${target.os}-${target.arch})...`,
  );
  const archive = await dependencies.download(releaseUrl(name, version));
  const checksums = await dependencies.download(
    releaseUrl(CHECKSUMS_NAME, version),
  );
  return installBinary(
    archive,
    checksums.toString("utf8"),
    name,
    binary,
    dependencies.extract,
  );
}

/** A child killed by a signal exits as 128 + N, the way shells report it. */
function exitCodeOf(result: RunResult): number {
  if (result.signal) {
    return 128 + (os.constants.signals[result.signal] ?? 0);
  }
  return result.status ?? 1;
}

/**
 * Run the wizard on this terminal and return its exit code.
 *
 * Ctrl+C reaches the whole foreground process group, and the wizard handles it
 * itself, so this process must not die while the child runs. A listener (not
 * an ignored disposition) is reset to the default in the child on exec.
 */
export function runBinary(
  binary: string,
  args: string[],
  run: LauncherDependencies["run"] = runInherited,
): number {
  const swallow = () => undefined;
  process.on("SIGINT", swallow);
  try {
    const result = run(binary, args);
    if (result.error) {
      throw new SetupLauncherError(
        `Could not start the setup wizard at ${binary}: ${result.error.message}`,
      );
    }
    return exitCodeOf(result);
  } finally {
    process.off("SIGINT", swallow);
  }
}

/** `deepeval setup [wizard options...]`, returning the exit code. */
export async function runSetup(
  args: string[],
  dependencies: LauncherDependencies = defaultLauncherDependencies(),
): Promise<number> {
  try {
    const binary = await ensureBinary(dependencies);
    return runBinary(binary, args, dependencies.run);
  } catch (error) {
    if (!(error instanceof SetupLauncherError)) throw error;
    dependencies.notify(error.message);
    return 1;
  }
}
