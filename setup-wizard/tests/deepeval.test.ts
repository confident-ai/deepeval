import { mkdtemp, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { describe, expect, it, vi } from "vitest";

import {
  describeCommands,
  evaluationTarget,
  inspectDeepEval,
  installDeepEval,
  nodeEnvironment,
  pythonEnvironments,
  type EnvironmentProbe,
} from "../src/deepeval.js";
import type { CommandRunner } from "../src/git.js";

const probe = (
  overrides: Partial<EnvironmentProbe> & { files?: Record<string, string> },
): EnvironmentProbe => {
  const files = overrides.files ?? {};
  return {
    exists: overrides.exists ?? (async (path) => path in files),
    readText: overrides.readText ?? (async (path) => files[path]),
    onPath: overrides.onPath ?? (async () => true),
    env: overrides.env ?? {},
  };
};

const ok: CommandRunner = async () => ({ stdout: "", stderr: "" });
const fails: CommandRunner = async () => {
  throw new Error("Command failed with exit code 1.");
};

describe("python environment detection", () => {
  it("prefers an active virtual environment over the project's own", async () => {
    const environments = await pythonEnvironments(
      "/app",
      probe({
        env: { VIRTUAL_ENV: "/tmp/venv" },
        files: {
          "/tmp/venv/bin/python": "",
          "/app/.venv/bin/python": "",
        },
      }),
    );
    expect(environments.map((environment) => environment.kind)).toEqual([
      "active-venv",
      "project-venv",
      "path-python",
    ]);
    expect(environments[0]?.install[0]?.command).toBe("/tmp/venv/bin/python");
  });

  it("uses each project manager's own install command", async () => {
    const poetry = await pythonEnvironments(
      "/app",
      probe({
        files: { "/app/pyproject.toml": "[tool.poetry]\nname = 'app'" },
      }),
    );
    expect(poetry[0]?.install).toEqual([
      { command: "poetry", args: ["add", "--group", "dev", "deepeval"] },
    ]);

    const uv = await pythonEnvironments(
      "/app",
      probe({ files: { "/app/uv.lock": "" } }),
    );
    expect(uv[0]?.install).toEqual([
      { command: "uv", args: ["add", "--dev", "deepeval"] },
    ]);
  });

  it("ignores a project manager that is not installed", async () => {
    const environments = await pythonEnvironments(
      "/app",
      probe({
        files: { "/app/pyproject.toml": "[tool.poetry]" },
        onPath: async () => false,
      }),
    );
    expect(environments.map((environment) => environment.kind)).toEqual([
      "path-python",
    ]);
  });
});

describe("deepeval preflight", () => {
  it("reports the environment that already has DeepEval", async () => {
    const status = await inspectDeepEval("/app", {
      probe: probe({ files: { "/app/.venv/bin/python": "" } }),
      runner: ok,
    });
    expect(status.installed?.kind).toBe("project-venv");
  });

  it("keeps looking past an environment that cannot import it", async () => {
    const runner = vi
      .fn<CommandRunner>()
      .mockRejectedValueOnce(new Error("no deepeval"))
      .mockResolvedValueOnce({ stdout: "", stderr: "" });
    const status = await inspectDeepEval("/app", {
      probe: probe({ files: { "/app/.venv/bin/python": "" } }),
      runner,
    });
    expect(status.installed?.kind).toBe("path-python");
    expect(runner).toHaveBeenCalledTimes(2);
  });

  it("never targets the system interpreter for a new install", async () => {
    const status = await inspectDeepEval("/app", {
      probe: probe({ files: { "/app/requirements.txt": "" } }),
      runner: fails,
    });
    expect(status.installed).toBeUndefined();
    expect(status.targets[0]?.kind).toBe("new-venv");
    expect(describeCommands(status.targets[0]?.install ?? [])).toBe(
      [
        "  python3 -m venv .venv",
        "  .venv/bin/python -m pip install deepeval",
      ].join("\n"),
    );
  });

  it("installs into an existing environment in command order", async () => {
    const runner = vi
      .fn<CommandRunner>()
      .mockResolvedValue({ stdout: "", stderr: "" });
    const status = await inspectDeepEval("/app", {
      probe: probe({ files: { "/app/.venv/bin/python": "" } }),
      runner: fails,
    });
    const [target] = status.targets;
    expect(target?.kind).toBe("project-venv");
    await installDeepEval(target!, "/app", runner);
    expect(runner).toHaveBeenCalledWith(
      ".venv/bin/python",
      ["-m", "pip", "install", "deepeval"],
      { cwd: "/app" },
    );
  });
});

describe("node environment detection", () => {
  const packageJson = (contents = "{}") => ({ "/app/package.json": contents });

  it("takes the package manager from a lockfile", async () => {
    const environment = await nodeEnvironment(
      "/app",
      probe({ files: { ...packageJson(), "/app/pnpm-lock.yaml": "" } }),
    );
    expect(environment?.kind).toBe("pnpm");
    expect(environment?.install).toEqual([
      { command: "pnpm", args: ["add", "deepeval"] },
    ]);
  });

  it("prefers the declared package manager over a stale lockfile", async () => {
    const environment = await nodeEnvironment(
      "/app",
      probe({
        files: {
          ...packageJson('{"packageManager":"yarn@4.9.1"}'),
          "/app/package-lock.json": "",
        },
      }),
    );
    expect(environment?.kind).toBe("yarn");
    expect(environment?.install).toEqual([
      { command: "yarn", args: ["add", "deepeval"] },
    ]);
  });

  it("falls back to the npm that ships with Node", async () => {
    const environment = await nodeEnvironment(
      "/app",
      probe({ files: packageJson() }),
    );
    expect(environment?.install).toEqual([
      { command: "npm", args: ["install", "deepeval"] },
    ]);
  });

  it("stays out of a project with no package manifest", async () => {
    expect(await nodeEnvironment("/app", probe({ files: {} }))).toBeUndefined();
  });

  it("reads an installed package off disk instead of resolving it", async () => {
    const runner = vi.fn<CommandRunner>().mockRejectedValue(new Error("no"));
    const status = await inspectDeepEval("/app", {
      probe: probe({
        files: {
          "/app/package.json": "{}",
          "/app/node_modules/deepeval/package.json": "{}",
        },
      }),
      runner,
    });
    expect(status.installed?.ecosystem).toBe("node");
  });

  /** Every other case stubs the probe, so nothing else covers the real one. */
  it("reads a real project directory through the default probe", async () => {
    const directory = await mkdtemp(join(tmpdir(), "deepeval-setup-"));
    await writeFile(
      join(directory, "package.json"),
      '{ "packageManager": "pnpm@10.4.1" }',
    );
    const environment = await nodeEnvironment(directory);
    expect(environment?.kind).toBe("pnpm");
    expect(environment?.check).toEqual({
      kind: "path",
      path: join("node_modules", "deepeval", "package.json"),
    });
  });

  it("offers one target per language when the repository holds both", async () => {
    const status = await inspectDeepEval("/app", {
      probe: probe({
        files: {
          "/app/pyproject.toml": "[project]",
          "/app/.venv/bin/python": "",
          "/app/package.json": "{}",
          "/app/bun.lock": "",
        },
      }),
      runner: fails,
    });
    expect(status.languages).toEqual(["python", "node"]);
    expect(status.targets.map(({ ecosystem }) => ecosystem)).toEqual([
      "python",
      "node",
    ]);
    expect(status.targets[1]?.install).toEqual([
      { command: "bun", args: ["add", "deepeval"] },
    ]);
  });
});

describe("a project in neither language", () => {
  const goProject = { "/app/go.mod": "module app", "/app/main.go": "" };

  it("reports no language and hosts the evaluation in Python", async () => {
    const status = await inspectDeepEval("/app", {
      probe: probe({ files: goProject }),
      runner: fails,
    });
    expect(status.languages).toEqual([]);
    expect(status.targets.map(({ kind }) => kind)).toEqual(["new-venv"]);
  });

  it("evaluates from the outside even when DeepEval is already importable", async () => {
    const status = await inspectDeepEval("/app", {
      probe: probe({ files: goProject }),
      runner: ok,
    });
    expect(status.installed?.kind).toBe("path-python");
    expect(
      evaluationTarget(status.languages, status.installed?.ecosystem),
    ).toEqual({ sdk: "python", shape: "black-box" });
  });

  it("does not read an active virtual environment as the project's language", async () => {
    const status = await inspectDeepEval("/app", {
      probe: probe({
        env: { VIRTUAL_ENV: "/tmp/venv" },
        files: { ...goProject, "/tmp/venv/bin/python": "" },
      }),
      runner: fails,
    });
    expect(status.languages).toEqual([]);
    expect(status.targets[0]?.kind).toBe("active-venv");
  });

  it("keeps a component-level shape for a project it can instrument", async () => {
    expect(evaluationTarget(["node"])).toEqual({
      sdk: "node",
      shape: "component-level",
    });
    expect(evaluationTarget(["python", "node"], "node")).toEqual({
      sdk: "node",
      shape: "component-level",
    });
  });
});
