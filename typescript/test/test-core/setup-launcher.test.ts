import { execFileSync } from "child_process";
import { createHash } from "crypto";
import * as fs from "fs";
import * as os from "os";
import * as path from "path";
import { Command } from "commander";

import { registerSetupCommand } from "@/cli/commands/setup";
import {
  cacheRoot,
  defaultLauncherDependencies,
  latestVersion,
  releaseUrl,
  RELEASES_PER_PAGE,
  resolveTarget,
  runSetup,
  type LauncherDependencies,
  type RunResult,
} from "@/cli/setup/launcher";

const ARCHIVE = "deepeval-setup-linux-arm64.tar.gz";
const BINARY = "#!/bin/sh\necho wizard\n";
const VERSION = "0.2.0";

let scratch: string;

beforeEach(() => {
  scratch = fs.mkdtempSync(path.join(os.tmpdir(), "deepeval-setup-test-"));
});

afterEach(() => {
  fs.rmSync(scratch, { recursive: true, force: true });
});

/** Builds a real tar.gz with the system `tar`, the same tool the launcher
 * extracts with. `link` makes the member a symlink instead of a file. */
function makeArchive(
  member: string,
  options: { link?: boolean; body?: string } = {},
): Buffer {
  const source = fs.mkdtempSync(path.join(scratch, "source-"));
  const file = path.join(source, member);
  if (options.link) {
    fs.symlinkSync("/etc/passwd", file);
  } else {
    fs.writeFileSync(file, options.body ?? BINARY, { mode: 0o644 });
  }
  const archive = path.join(scratch, `archive-${path.basename(source)}.tgz`);
  execFileSync("tar", ["-czf", archive, "-C", source, member]);
  return fs.readFileSync(archive);
}

function checksums(archive: Buffer, name = ARCHIVE): Buffer {
  const digest = createHash("sha256").update(archive).digest("hex");
  return Buffer.from(`${"0".repeat(64)}  other.tar.gz\n${digest}  ${name}\n`);
}

function githubRelease(
  tag: string,
  options: { assets?: string[]; draft?: boolean; prerelease?: boolean } = {},
) {
  return {
    tag_name: tag,
    draft: options.draft ?? false,
    prerelease: options.prerelease ?? false,
    assets: (options.assets ?? [ARCHIVE, "SHA256SUMS"]).map((name) => ({
      name,
    })),
  };
}

/** Stands in for GitHub releases and the child process, so no test touches
 * the network or launches anything. */
function fakeDependencies(
  files: Record<string, Buffer>,
  result: RunResult = { status: 0, signal: null },
  overrides: Partial<LauncherDependencies> = {},
  pages: ReadonlyArray<ReadonlyArray<unknown>> = [
    [githubRelease("python-v4.3.0"), githubRelease(`setup-v${VERSION}`)],
  ],
) {
  const requested: string[] = [];
  const pagesRequested: number[] = [];
  const runs: Array<{ binary: string; args: string[] }> = [];
  const messages: string[] = [];
  const dependencies: LauncherDependencies = {
    fetchReleases: async (page) => {
      pagesRequested.push(page);
      return [...(pages[page - 1] ?? [])];
    },
    download: async (url) => {
      requested.push(url);
      const body = files[url];
      if (!body) throw new Error(`unexpected download ${url}`);
      return body;
    },
    // The real system `tar` call, so its flags are exercised too.
    extract: defaultLauncherDependencies().extract,
    run: (binary, args) => {
      runs.push({ binary, args });
      return result;
    },
    platform: "linux",
    arch: "arm64",
    env: { XDG_CACHE_HOME: path.join(scratch, "cache") },
    home: path.join(scratch, "home"),
    notify: (message) => messages.push(message),
    ...overrides,
  };
  return { dependencies, requested, pagesRequested, runs, messages };
}

function release(archive: Buffer, sums: Buffer = checksums(archive)) {
  return {
    [releaseUrl(ARCHIVE, VERSION)]: archive,
    [releaseUrl("SHA256SUMS", VERSION)]: sums,
  };
}

const binaryPath = (version = VERSION) =>
  path.join(scratch, "cache", "deepeval", "setup", version, "deepeval-setup");

function cacheBinary(version = VERSION): string {
  const binary = binaryPath(version);
  fs.mkdirSync(path.dirname(binary), { recursive: true });
  fs.writeFileSync(binary, BINARY, { mode: 0o755 });
  return binary;
}

describe("deepeval setup launcher", () => {
  it("downloads, verifies, caches and runs the latest wizard", async () => {
    const archive = makeArchive("deepeval-setup");
    const fake = fakeDependencies(release(archive), {
      status: 7,
      signal: null,
    });

    await expect(runSetup(["--proj-id", "p"], fake.dependencies)).resolves.toBe(
      7,
    );

    const binary = binaryPath();
    expect(fs.readFileSync(binary, "utf8")).toBe(BINARY);
    expect(fs.statSync(binary).mode & 0o777).toBe(0o755);
    expect(fake.requested).toEqual([
      `https://github.com/confident-ai/deepeval/releases/download/setup-v${VERSION}/${ARCHIVE}`,
      `https://github.com/confident-ai/deepeval/releases/download/setup-v${VERSION}/SHA256SUMS`,
    ]);
    expect(fake.runs).toEqual([{ binary, args: ["--proj-id", "p"] }]);
    // Only the binary lands in the cache: the staging directory is gone.
    expect(fs.readdirSync(path.dirname(binary))).toEqual(["deepeval-setup"]);
  });

  it("picks the newest published release that ships this platform", async () => {
    const pages = [
      [
        githubRelease("typescript-v0.9.21"),
        githubRelease("setup-v0.4.0", { draft: true }),
        githubRelease("setup-v0.3.5", { prerelease: true }),
        githubRelease("setup-v0.3.0", { assets: ["SHA256SUMS"] }),
        githubRelease("setup-v0.10.0"),
        githubRelease("setup-v0.9.0"),
        githubRelease("setup-latest"),
        "not a release",
      ],
    ];
    const fetched: number[] = [];

    await expect(
      latestVersion(ARCHIVE, async (page) => {
        fetched.push(page);
        return pages[page - 1] ?? [];
      }),
    ).resolves.toBe("0.10.0");
    expect(fetched).toEqual([1]);
  });

  it("looks past a full page without a wizard release", async () => {
    const fullPage = Array.from({ length: RELEASES_PER_PAGE }, (_, index) =>
      githubRelease(`python-v4.${index}.0`),
    );
    const pages = [fullPage, [githubRelease("setup-v0.1.0")]];

    await expect(
      latestVersion(ARCHIVE, async (page) => pages[page - 1] ?? []),
    ).resolves.toBe("0.1.0");
  });

  it("runs a cached latest wizard without downloading", async () => {
    const binary = cacheBinary();
    const fake = fakeDependencies({});

    await expect(
      runSetup(["--org-id", "org"], fake.dependencies),
    ).resolves.toBe(0);

    expect(fake.requested).toEqual([]);
    expect(fake.runs).toEqual([{ binary, args: ["--org-id", "org"] }]);
  });

  it("replaces an older cached wizard with a newer release", async () => {
    cacheBinary("0.1.0");
    const archive = makeArchive("deepeval-setup");
    const fake = fakeDependencies(release(archive));

    await expect(runSetup([], fake.dependencies)).resolves.toBe(0);

    expect(fake.requested).toHaveLength(2);
    expect(fake.runs).toEqual([{ binary: binaryPath(), args: [] }]);
  });

  it("falls back to the newest cached wizard when GitHub is down", async () => {
    cacheBinary("0.1.0");
    const newest = cacheBinary("0.10.0");
    const fake = fakeDependencies({}, undefined, {
      fetchReleases: async () => {
        throw new Error("offline");
      },
    });

    await expect(runSetup([], fake.dependencies)).resolves.toBe(0);

    expect(fake.messages.join("\n")).toContain("using the cached 0.10.0");
    expect(fake.runs).toEqual([{ binary: newest, args: [] }]);
  });

  it.each([
    ["no releases", []],
    ["no wizard release", [[githubRelease("python-v4.3.0")]]],
  ] as const)("exits 1 with %s and nothing cached", async (_label, pages) => {
    const fake = fakeDependencies({}, undefined, {}, pages);

    await expect(runSetup([], fake.dependencies)).resolves.toBe(1);

    expect(fake.messages.join("\n")).toContain(
      "Could not find the setup wizard",
    );
    expect(fake.runs).toEqual([]);
  });

  it("writes nothing when the checksum does not match", async () => {
    const archive = makeArchive("deepeval-setup");
    const tampered = makeArchive("deepeval-setup", { body: "echo evil\n" });
    const fake = fakeDependencies(release(tampered, checksums(archive)));

    await expect(runSetup([], fake.dependencies)).resolves.toBe(1);

    expect(fake.messages.join("\n")).toContain("Checksum mismatch");
    expect(fs.existsSync(path.join(scratch, "cache"))).toBe(false);
    expect(fake.runs).toEqual([]);
  });

  it("rejects an archive whose binary is not a regular file", async () => {
    const archive = makeArchive("deepeval-setup", { link: true });
    const fake = fakeDependencies(release(archive));

    await expect(runSetup([], fake.dependencies)).resolves.toBe(1);

    expect(fake.messages.join("\n")).toContain("not a regular file");
    expect(fs.existsSync(binaryPath())).toBe(false);
    expect(fake.runs).toEqual([]);
  });

  it("rejects an archive without the binary", async () => {
    const archive = makeArchive("other");
    const fake = fakeDependencies(release(archive), undefined, {
      // System tar fails on a missing member; this proves the launcher's own
      // check too, whatever the tar implementation does.
      extract: async () => undefined,
    });

    await expect(runSetup([], fake.dependencies)).resolves.toBe(1);

    expect(fake.messages.join("\n")).toContain("does not contain");
    expect(fs.existsSync(binaryPath())).toBe(false);
  });

  it.each([
    ["win32", "x64"],
    ["linux", "ppc64"],
    ["darwin", "ia32"],
  ] as const)("exits 1 on %s/%s", async (platform, arch) => {
    const fake = fakeDependencies({}, undefined, { platform, arch });

    await expect(runSetup([], fake.dependencies)).resolves.toBe(1);

    expect(fake.messages.join("\n")).toContain("macOS and Linux");
    expect(fake.pagesRequested).toEqual([]);
    expect(fake.requested).toEqual([]);
    expect(fake.runs).toEqual([]);
  });

  it("maps the supported platforms", () => {
    expect(resolveTarget("darwin", "x64")).toEqual({
      os: "darwin",
      arch: "x64",
    });
    expect(resolveTarget("darwin", "arm64")).toEqual({
      os: "darwin",
      arch: "arm64",
    });
    expect(resolveTarget("linux", "x64")).toEqual({ os: "linux", arch: "x64" });
    expect(resolveTarget("linux", "arm64")).toEqual({
      os: "linux",
      arch: "arm64",
    });
  });

  it("reports a signal exit the way shells do", async () => {
    cacheBinary();
    const fake = fakeDependencies({}, { status: null, signal: "SIGINT" });

    await expect(runSetup([], fake.dependencies)).resolves.toBe(130);
  });

  it("ignores a relative XDG_CACHE_HOME", () => {
    expect(cacheRoot({ XDG_CACHE_HOME: "relative" }, "/home/me")).toBe(
      path.join("/home/me", ".cache", "deepeval", "setup"),
    );
  });

  it("passes every wizard option through the CLI", async () => {
    const binary = cacheBinary();
    const fake = fakeDependencies({}, { status: 3, signal: null });
    const program = new Command().enablePositionalOptions().exitOverride();
    registerSetupCommand(program, () => fake.dependencies);
    const previousExitCode = process.exitCode;

    try {
      await program.parseAsync(
        [
          "setup",
          "--project-dir",
          "app",
          "--proj-id=project",
          "--from",
          "deepeval",
          "-h",
        ],
        { from: "user" },
      );
      expect(process.exitCode).toBe(3);
    } finally {
      process.exitCode = previousExitCode;
    }

    expect(fake.runs).toEqual([
      {
        binary,
        args: [
          "--project-dir",
          "app",
          "--proj-id=project",
          "--from",
          "deepeval",
          "-h",
        ],
      },
    ]);
  });
});
