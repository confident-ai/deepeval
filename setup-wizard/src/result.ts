import { readFile } from "node:fs/promises";

import { z } from "zod";

export const setupResultSchema = z
  .object({
    status: z.enum(["completed", "partial", "failed"]),
    /**
     * Component-level is the shape the wizard asks for. `black-box` is the
     * fallback an application in neither SDK's language has to fall back to,
     * and it has no spans to report, so the two are validated apart.
     */
    shape: z.enum(["component-level", "black-box"]).default("component-level"),
    changedFiles: z.array(z.string().min(1)),
    sdks: z
      .array(z.enum(["deepeval-python", "deepeval-typescript"]))
      .min(1)
      .max(2),
    levels: z.array(z.enum(["test-case", "span", "trace", "thread"])).min(1),
    datasetSource: z.string().min(1),
    metrics: z.array(z.string().min(1)).max(12),
    rerunCommand: z.string().min(1),
    testRunId: z.string().min(1).optional(),
    testRunUrl: z.string().url().optional(),
    errors: z.array(z.string().min(1)).optional(),
  })
  .strict()
  .superRefine((result, context) => {
    if (result.status === "failed") {
      if (!result.errors?.length) {
        context.addIssue({
          code: "custom",
          path: ["errors"],
          message: "A failed setup must include at least one error.",
        });
      }
      return;
    }
    /**
     * Component metrics are the evaluation the wizard asks for, and a span is
     * where one can live, so a suite reporting none of them built something
     * else. The trace carries the run-wide metrics, task completion among them,
     * which is what makes the component scores legible.
     */
    if (result.shape === "component-level") {
      for (const level of ["span", "trace"] as const) {
        if (result.levels.includes(level)) continue;
        context.addIssue({
          code: "custom",
          path: ["levels"],
          message: `A component-level evaluation must report the ${level} level. Report the black-box shape when the application has no SDK to instrument it with, or failed when the suite could not reach that level.`,
        });
      }
    }
    /** Spans come from instrumentation, which is what black-box rules out. */
    if (result.shape === "black-box" && result.levels.includes("span")) {
      context.addIssue({
        code: "custom",
        path: ["levels"],
        message:
          "A black-box evaluation cannot report the span level. Report the component-level shape when the application was instrumented.",
      });
    }
  });

export type SetupResult = z.infer<typeof setupResultSchema>;

export const parseSetupResult = (
  value: unknown,
  requireTestRun = true,
): SetupResult => {
  const result = setupResultSchema.parse(value);
  if (requireTestRun && result.status === "completed" && !result.testRunId) {
    throw new z.ZodError([
      {
        code: "custom",
        path: ["testRunId"],
        message: "A completed setup must include testRunId.",
      },
    ]);
  }
  return result;
};

/**
 * A file the contract rejects was still written after the suite ran, so the
 * test run it names may exist and its rerun command may work. Read those few
 * fields apart from the contract so a reporting mistake never costs the user
 * the evaluation the agent already spent judge money on.
 */
const salvageSchema = z
  .object({
    rerunCommand: z.string().min(1).optional(),
    testRunId: z.string().min(1).optional(),
    testRunUrl: z.string().min(1).optional(),
  })
  .loose();

export type SalvagedResult = z.infer<typeof salvageSchema>;

export interface RejectedSetupResult {
  issues: string[];
  salvaged: SalvagedResult;
}

export type SetupResultOutcome =
  | { ok: true; result: SetupResult }
  | { ok: false; rejected: RejectedSetupResult };

const describeIssues = (error: unknown): string[] => {
  if (error instanceof z.ZodError) {
    return error.issues.map((issue) => {
      const path = issue.path.join(".");
      return path ? `${path}: ${issue.message}` : issue.message;
    });
  }
  return [error instanceof Error ? error.message : String(error)];
};

export const readSetupResult = async (
  path: string,
  requireTestRun = true,
): Promise<SetupResultOutcome> => {
  let value: unknown;
  try {
    value = JSON.parse(await readFile(path, "utf8"));
  } catch (error) {
    return {
      ok: false,
      rejected: { issues: describeIssues(error), salvaged: {} },
    };
  }
  try {
    return { ok: true, result: parseSetupResult(value, requireTestRun) };
  } catch (error) {
    const salvaged = salvageSchema.safeParse(value);
    return {
      ok: false,
      rejected: {
        issues: describeIssues(error),
        salvaged: salvaged.success ? salvaged.data : {},
      },
    };
  }
};
