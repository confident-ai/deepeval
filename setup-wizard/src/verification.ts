import { z } from "zod";

export const testRunStatuses = [
  "IN_PROGRESS",
  "COMPLETED",
  "ERRORED",
  "CANCELLED",
] as const;

const testRunSchema = z.object({
  data: z
    .object({
      status: z.enum(testRunStatuses).optional(),
      totalTests: z.number().optional(),
      testsPassed: z.number().optional(),
      testsFailed: z.number().optional(),
    })
    .optional(),
});

export type TestRunStatus = (typeof testRunStatuses)[number];

export interface VerificationResult {
  testRunId: string;
  /** Absent when the project is unknown, as it is for a reused key. */
  testRunUrl?: string;
  status?: TestRunStatus;
}

export const buildTestRunUrl = (
  appUrl: string,
  projectId: string,
  testRunId: string,
): string =>
  `${appUrl.replace(/\/+$/, "")}/project/${encodeURIComponent(projectId)}/test-runs/${encodeURIComponent(testRunId)}`;

export const verifyTestRun = async (
  apiUrl: string,
  appUrl: string,
  apiKey: string,
  projectId: string | undefined,
  testRunId: string,
  fetchImplementation: typeof globalThis.fetch = globalThis.fetch,
): Promise<VerificationResult> => {
  const id = z.string().trim().min(1).parse(testRunId);
  const response = await fetchImplementation(
    `${apiUrl.replace(/\/+$/, "")}/v1/test-runs/${encodeURIComponent(id)}`,
    { headers: { CONFIDENT_API_KEY: apiKey } },
  );
  if (!response.ok) {
    throw new Error(
      `Test run ${id} could not be verified (${response.status}).`,
    );
  }

  const body: unknown = await response.json().catch(() => ({}));
  const status = testRunSchema.safeParse(body).data?.data?.status;
  if (status === "ERRORED" || status === "CANCELLED") {
    throw new Error(`Test run ${id} finished with status ${status}.`);
  }

  return {
    testRunId: id,
    ...(projectId
      ? { testRunUrl: buildTestRunUrl(appUrl, projectId, id) }
      : {}),
    ...(status ? { status } : {}),
  };
};
