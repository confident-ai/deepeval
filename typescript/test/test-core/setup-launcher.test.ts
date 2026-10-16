import { execFileSync } from "child_process";
import { createHash } from "crypto";
import * as fs from "fs";
import * as os from "os";
import * as path from "path";
import { Command } from "commander";

import { registerSetupCommand } from "@/cli/commands/setup";
import {
  cacheDir,
  defaultLauncherDependencies,
  releaseUrl,
  resolveTarget,
  runSetup,
  SETUP_WIZARD_VERSION,
  type LauncherDependencies,
  type RunResult,
} from "@/cli/setup/launcher";

const ARCHIVE = "deepeval-setup-linux-arm64.tar.gz";
const BINARY = "#!/bin/sh\necho wizard\n";

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

/** Stands in for GitHub releases and the child process, so no test touches
 * the network or launches anything. */
function fakeDependencies(
  files: Record<string, Buffer>,
  result: RunResult = { status: 0, signal: null },
  overrides: Partial<LauncherDependencies> = {},
) {
  const requested: string[] = [];
  const runs: Array<{ binary: string; args: string[] }> = [];
  const messages: string[] = [];
  const dependencies: LauncherDependencies = {
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
  return { dependencies, requested, runs, messages };
}

function release(archive: Buffer, sums: Buffer = checksums(archive)) {
  return {
    [releaseUrl(ARCHIVE)]: archive,
    [releaseUrl("SHA256SUMS")]: sums,
  };
}

const binaryPath = () =>
  path.join(
    scratch,
    "cache",
    "deepeval",
    "setup",
    SETUP_WIZARD_VERSION,
    "deepeval-setup",
  );

describe("deepeval setup launcher", () => {
  it("downloads, verifies, caches and runs the wizard", async () => {
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
      `https://github.com/confident-ai/deepeval/releases/download/setup-v${SETUP_WIZARD_VERSION}/${ARCHIVE}`,
      `https://github.com/confident-ai/deepeval/releases/download/setup-v${SETUP_WIZARD_VERSION}/SHA256SUMS`,
    ]);
    expect(fake.runs).toEqual([{ binary, args: ["--proj-id", "p"] }]);
    // Only the binary lands in the cache: the staging directory is gone.
    expect(fs.readdirSync(path.dirname(binary))).toEqual(["deepeval-setup"]);
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

  it("runs a cached binary without downloading", async () => {
    const binary = binaryPath();
    fs.mkdirSync(path.dirname(binary), { recursive: true });
    fs.writeFileSync(binary, BINARY, { mode: 0o755 });
    const fake = fakeDependencies({});

    await expect(
      runSetup(["--org-id", "org"], fake.dependencies),
    ).resolves.toBe(0);

    expect(fake.requested).toEqual([]);
    expect(fake.runs).toEqual([{ binary, args: ["--org-id", "org"] }]);
  });

  it("reports a signal exit the way shells do", async () => {
    const binary = binaryPath();
    fs.mkdirSync(path.dirname(binary), { recursive: true });
    fs.writeFileSync(binary, BINARY, { mode: 0o755 });
    const fake = fakeDependencies({}, { status: null, signal: "SIGINT" });

    await expect(runSetup([], fake.dependencies)).resolves.toBe(130);
  });

  it("ignores a relative XDG_CACHE_HOME", () => {
    expect(cacheDir({ XDG_CACHE_HOME: "relative" }, "/home/me")).toBe(
      path.join(
        "/home/me",
        ".cache",
        "deepeval",
        "setup",
        SETUP_WIZARD_VERSION,
      ),
    );
  });

  it("passes every wizard option through the CLI", async () => {
    const binary = binaryPath();
    fs.mkdirSync(path.dirname(binary), { recursive: true });
    fs.writeFileSync(binary, BINARY, { mode: 0o755 });
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

  it("pins the version of setup-wizard/package.json", () => {
    const manifest = JSON.parse(
      fs.readFileSync(
        path.resolve(__dirname, "../../../setup-wizard/package.json"),
        "utf8",
      ),
    ) as { version: string };
    expect(SETUP_WIZARD_VERSION).toBe(manifest.version);
  });
});
