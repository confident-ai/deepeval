import { constants } from "node:fs";
import { access, readFile, stat } from "node:fs/promises";
import { delimiter, join } from "node:path";

import { runCommand, type CommandRunner } from "./git.js";

export const DEEPEVAL_PACKAGE = "deepeval";

/** Cheaper than importing DeepEval, which pulls in its whole metric tree. */
const IMPORT_CHECK =
  "import importlib.util as u, sys; sys.exit(0 if u.find_spec('deepeval') else 1)";

export interface Command {
  command: string;
  args: string[];
}

/** DeepEval ships one SDK per language, so the project decides which to set up. */
export type Ecosystem = "python" | "node";

export type EnvironmentKind =
  | "active-venv"
  | "poetry"
  | "uv"
  | "project-venv"
  | "path-python"
  | "new-venv"
  | "npm"
  | "pnpm"
  | "yarn"
  | "bun";

/**
 * A Python interpreter answers whether it can import DeepEval, but asking a
 * package manager the same question means running an install-time resolver, so
 * the Node ecosystem checks for the installed package on disk instead.
 */
export type EnvironmentCheck =
  { kind: "command"; run: Command } | { kind: "path"; path: string };

export interface DeepEvalEnvironment {
  ecosystem: Ecosystem;
  kind: EnvironmentKind;
  /** Names where DeepEval lives or would land, for the confirmation prompt. */
  label: string;
  check: EnvironmentCheck;
  /** Run in order; the last one installs DeepEval. */
  install: Command[];
}

export interface EnvironmentProbe {
  exists: (path: string) => Promise<boolean>;
  readText: (path: string) => Promise<string | undefined>;
  onPath: (command: string) => Promise<boolean>;
  env: NodeJS.ProcessEnv;
}

const pathExists = async (path: string): Promise<boolean> => {
  try {
    await access(path);
    return true;
  } catch {
    return false;
  }
};

const readTextFile = async (path: string): Promise<string | undefined> => {
  try {
    return await readFile(path, "utf8");
  } catch {
    return undefined;
  }
};

/** `command -v` is a shell builtin that most Linux images do not ship as a
 * binary, so PATH is walked directly instead. */
const onPath = async (command: string): Promise<boolean> => {
  if (command.includes("/")) return pathExists(command);
  for (const entry of (process.env.PATH ?? "").split(delimiter)) {
    if (!entry) continue;
    const candidate = join(entry, command);
    try {
      if (!(await stat(candidate)).isFile()) continue;
      await access(candidate, constants.X_OK);
      return true;
    } catch {
      // Try the next PATH entry.
    }
  }
  return false;
};

export const defaultProbe: EnvironmentProbe = {
  exists: pathExists,
  readText: readTextFile,
  onPath,
  env: process.env,
};

const pythonImportCheck = (python: string): EnvironmentCheck => ({
  kind: "command",
  run: { command: python, args: ["-c", IMPORT_CHECK] },
});

const commandCheck = (command: string, args: string[]): EnvironmentCheck => ({
  kind: "command",
  run: { command, args },
});

const pipInstall = (python: string): Command => ({
  command: python,
  args: ["-m", "pip", "install", DEEPEVAL_PACKAGE],
});

/** Relative so the confirmation prompt shows a command worth reading. */
const PROJECT_VENV_PYTHON = join(".venv", "bin", "python");

/**
 * Ordered by how likely each environment is the one the evaluation will run
 * in, so the first that already has DeepEval wins and the first that could
 * hold it becomes the install target.
 */
export const pythonEnvironments = async (
  projectDirectory: string,
  probe: EnvironmentProbe = defaultProbe,
): Promise<DeepEvalEnvironment[]> => {
  const environments: DeepEvalEnvironment[] = [];

  const activeVenv = probe.env.VIRTUAL_ENV?.trim();
  if (activeVenv) {
    const python = join(activeVenv, "bin", "python");
    if (await probe.exists(python)) {
      environments.push({
        ecosystem: "python",
        kind: "active-venv",
        label: "the active virtual environment",
        check: pythonImportCheck(python),
        install: [pipInstall(python)],
      });
    }
  }

  const manifest = await probe.readText(
    join(projectDirectory, "pyproject.toml"),
  );
  if (manifest?.includes("[tool.poetry") && (await probe.onPath("poetry"))) {
    environments.push({
      ecosystem: "python",
      kind: "poetry",
      label: "this project's Poetry environment",
      check: commandCheck("poetry", ["run", "python", "-c", IMPORT_CHECK]),
      install: [
        {
          command: "poetry",
          args: ["add", "--group", "dev", DEEPEVAL_PACKAGE],
        },
      ],
    });
  }
  const usesUv =
    manifest?.includes("[tool.uv") ||
    (await probe.exists(join(projectDirectory, "uv.lock")));
  if (usesUv && (await probe.onPath("uv"))) {
    environments.push({
      ecosystem: "python",
      kind: "uv",
      label: "this project's uv environment",
      check: commandCheck("uv", ["run", "python", "-c", IMPORT_CHECK]),
      install: [{ command: "uv", args: ["add", "--dev", DEEPEVAL_PACKAGE] }],
    });
  }

  if (await probe.exists(join(projectDirectory, PROJECT_VENV_PYTHON))) {
    environments.push({
      ecosystem: "python",
      kind: "project-venv",
      label: "the project's .venv",
      check: pythonImportCheck(PROJECT_VENV_PYTHON),
      install: [pipInstall(PROJECT_VENV_PYTHON)],
    });
  }

  environments.push({
    ecosystem: "python",
    kind: "path-python",
    label: "the python3 on your PATH",
    check: pythonImportCheck("python3"),
    install: [pipInstall("python3")],
  });

  return environments;
};

/**
 * A bare `python3` may already carry DeepEval, which is worth honoring, but it
 * is never a good place to install one: a system interpreter is often managed
 * externally and rejects pip outright.
 */
export const newVenvEnvironment = (): DeepEvalEnvironment => ({
  ecosystem: "python",
  kind: "new-venv",
  label: "a new .venv in this project",
  check: pythonImportCheck(PROJECT_VENV_PYTHON),
  install: [
    { command: "python3", args: ["-m", "venv", ".venv"] },
    pipInstall(PROJECT_VENV_PYTHON),
  ],
});

/**
 * Every manager here writes to `dependencies` by default, which is what the
 * evaluation needs: component-level instrumentation puts `deepeval/tracing`
 * imports in application code, not only in the eval suite.
 */
const NODE_MANAGERS: {
  kind: EnvironmentKind & ("npm" | "pnpm" | "yarn" | "bun");
  lockfiles: string[];
  install: Command;
}[] = [
  {
    kind: "pnpm",
    lockfiles: ["pnpm-lock.yaml"],
    install: { command: "pnpm", args: ["add", DEEPEVAL_PACKAGE] },
  },
  {
    kind: "yarn",
    lockfiles: ["yarn.lock"],
    install: { command: "yarn", args: ["add", DEEPEVAL_PACKAGE] },
  },
  {
    kind: "bun",
    lockfiles: ["bun.lock", "bun.lockb"],
    install: { command: "bun", args: ["add", DEEPEVAL_PACKAGE] },
  },
  {
    kind: "npm",
    lockfiles: ["package-lock.json"],
    install: { command: "npm", args: ["install", DEEPEVAL_PACKAGE] },
  },
];

/** Where a manager's own install puts the package, so no resolver has to run. */
const NODE_PACKAGE_MANIFEST = join(
  "node_modules",
  DEEPEVAL_PACKAGE,
  "package.json",
);

/**
 * A single environment, because a Node project has exactly one package manager:
 * the `packageManager` field when Corepack declares one, else the lockfile,
 * else npm, which ships with Node itself.
 */
export const nodeEnvironment = async (
  projectDirectory: string,
  probe: EnvironmentProbe = defaultProbe,
): Promise<DeepEvalEnvironment | undefined> => {
  const manifest = await probe.readText(join(projectDirectory, "package.json"));
  if (manifest === undefined) return undefined;

  const declared = /"packageManager"\s*:\s*"([a-z]+)@/.exec(manifest)?.[1];
  const byDeclaration = NODE_MANAGERS.find(({ kind }) => kind === declared);
  let manager = byDeclaration;
  if (!manager) {
    for (const candidate of NODE_MANAGERS) {
      const present = await Promise.all(
        candidate.lockfiles.map((lockfile) =>
          probe.exists(join(projectDirectory, lockfile)),
        ),
      );
      if (present.some(Boolean)) {
        manager = candidate;
        break;
      }
    }
  }
  const { kind, install } = manager ?? NODE_MANAGERS[NODE_MANAGERS.length - 1]!;

  return {
    ecosystem: "node",
    kind,
    label: `this project's ${kind} dependencies`,
    check: { kind: "path", path: NODE_PACKAGE_MANIFEST },
    install: [install],
  };
};

export interface DeepEvalStatus {
  /** Set when some environment already has DeepEval. */
  installed?: DeepEvalEnvironment;
  /**
   * Where an install could go when nothing has it yet, best first. A project
   * that carries both a Python and a Node manifest gets one target each, since
   * only the user knows which half the agent should evaluate.
   */
  targets: DeepEvalEnvironment[];
  /**
   * What the project itself is written in. Empty means neither language, so
   * DeepEval cannot instrument this application and can only evaluate it from
   * the outside, from a script the wizard's own Python target hosts.
   */
  languages: Ecosystem[];
}

/**
 * `component-level` instruments the application and scores its spans.
 * `black-box` is the fallback for an application DeepEval has no SDK for: it
 * calls the application the way a user would and scores test cases built from
 * goldens, with no spans to attach anything to.
 */
export type EvaluationShape = "component-level" | "black-box";

export interface EvaluationTarget {
  sdk: Ecosystem;
  shape: EvaluationShape;
}

export const describeCommands = (commands: Command[]): string =>
  commands
    .map(({ command, args }) => `  ${[command, ...args].join(" ")}`)
    .join("\n");

export interface DeepEvalDependencies {
  probe?: EnvironmentProbe;
  runner?: CommandRunner;
}

/** A Python manifest is what makes a repository a Python project to install into. */
const PYTHON_MANIFESTS = [
  "pyproject.toml",
  "requirements.txt",
  "setup.py",
  "setup.cfg",
  "Pipfile",
];

const hasPythonManifest = async (
  projectDirectory: string,
  probe: EnvironmentProbe,
): Promise<boolean> => {
  for (const manifest of PYTHON_MANIFESTS) {
    if (await probe.exists(join(projectDirectory, manifest))) return true;
  }
  return false;
};

const alreadyInstalled = async (
  environment: DeepEvalEnvironment,
  projectDirectory: string,
  probe: EnvironmentProbe,
  runner: CommandRunner,
): Promise<boolean> => {
  if (environment.check.kind === "path") {
    return probe.exists(join(projectDirectory, environment.check.path));
  }
  try {
    await runner(environment.check.run.command, environment.check.run.args, {
      cwd: projectDirectory,
    });
    return true;
  } catch {
    // A missing interpreter reads the same as a missing import: keep looking.
    return false;
  }
};

export const inspectDeepEval = async (
  projectDirectory: string,
  { probe = defaultProbe, runner = runCommand }: DeepEvalDependencies = {},
): Promise<DeepEvalStatus> => {
  const node = await nodeEnvironment(projectDirectory, probe);
  const python = await pythonEnvironments(projectDirectory, probe);

  /**
   * Evidence that the project is Python, which an interpreter alone is not: a
   * `python3` is on every machine and a virtual environment can be active in
   * any directory, so only a manifest or the project's own `.venv` counts.
   */
  const languages: Ecosystem[] = [];
  if (
    (await hasPythonManifest(projectDirectory, probe)) ||
    python.some(({ kind }) => kind === "project-venv")
  ) {
    languages.push("python");
  }
  if (node) languages.push("node");

  /**
   * A Python interpreter also hosts the black-box script, so it is searched
   * either way; a Node environment only matters to a Node project.
   */
  const candidates = [
    ...python,
    ...(node && languages.includes("node") ? [node] : []),
  ];
  for (const environment of candidates) {
    if (await alreadyInstalled(environment, projectDirectory, probe, runner)) {
      return { installed: environment, targets: [environment], languages };
    }
  }

  const targets: DeepEvalEnvironment[] = [];
  const installablePython = python.find(({ kind }) => kind !== "path-python");
  if (languages.includes("python")) {
    targets.push(installablePython ?? newVenvEnvironment());
  }
  if (node && languages.includes("node")) targets.push(node);
  /** Neither language: Python hosts the evaluation script rather than the app. */
  if (targets.length === 0) {
    targets.push(installablePython ?? newVenvEnvironment());
  }
  return { targets, languages };
};

/**
 * The SDK to write in, and whether it can reach inside the application. A
 * project in neither language still evaluates, from the outside.
 */
export const evaluationTarget = (
  languages: Ecosystem[],
  chosen?: Ecosystem,
): EvaluationTarget =>
  languages.length === 0
    ? { sdk: "python", shape: "black-box" }
    : {
        sdk: chosen && languages.includes(chosen) ? chosen : languages[0]!,
        shape: "component-level",
      };

export const installDeepEval = async (
  environment: DeepEvalEnvironment,
  projectDirectory: string,
  runner: CommandRunner = runCommand,
): Promise<void> => {
  for (const { command, args } of environment.install) {
    await runner(command, args, { cwd: projectDirectory });
  }
};
