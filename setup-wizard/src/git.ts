import { execFile } from "node:child_process";
import { promisify } from "node:util";

const execFileAsync = promisify(execFile);

export interface GitStatus {
  isRepository: boolean;
  dirty: boolean;
  files: string[];
  totalChanges: number;
}

export type CommandRunner = (
  command: string,
  args: string[],
  options: { cwd: string },
) => Promise<{ stdout: string; stderr: string }>;

export const runCommand: CommandRunner = async (command, args, options) => {
  const result = await execFileAsync(command, args, {
    cwd: options.cwd,
    encoding: "utf8",
  });
  return { stdout: result.stdout, stderr: result.stderr };
};

export const inspectGit = async (
  cwd: string,
  runner: CommandRunner = runCommand,
): Promise<GitStatus> => {
  try {
    const repository = await runner(
      "git",
      ["rev-parse", "--is-inside-work-tree"],
      { cwd },
    );
    if (repository.stdout.trim() !== "true") {
      return {
        isRepository: false,
        dirty: false,
        files: [],
        totalChanges: 0,
      };
    }
  } catch {
    return {
      isRepository: false,
      dirty: false,
      files: [],
      totalChanges: 0,
    };
  }

  const status = await runner(
    "git",
    ["status", "--short", "--untracked-files=all"],
    { cwd },
  );
  const allFiles = status.stdout
    .split(/\r?\n/)
    .map((line) => line.trimEnd())
    .filter(Boolean);

  return {
    isRepository: true,
    dirty: allFiles.length > 0,
    files: allFiles.slice(0, 20),
    totalChanges: allFiles.length,
  };
};

export const describeGitStatus = (status: GitStatus): string => {
  if (!status.isRepository) {
    return "The wizard may edit files without Git history to help review them.";
  }
  if (!status.dirty) return "Git working tree is clean.";
  const remainder = status.totalChanges - status.files.length;
  return [
    ...status.files.map(
      (file, index) =>
        `${index + 1}. ${file.length > 3 ? file.slice(3) : file}`,
    ),
    ...(remainder > 0 ? [`…plus ${remainder} more`] : []),
  ].join("\n");
};
