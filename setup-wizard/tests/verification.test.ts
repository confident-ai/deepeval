import { describe, expect, it, vi } from "vitest";

import { buildTestRunUrl, verifyTestRun } from "../src/verification.js";

describe("test run verification", () => {
  it("authenticates the API lookup and returns the app permalink", async () => {
    const fetch = vi
      .fn<typeof globalThis.fetch>()
      .mockResolvedValue(new Response("{}", { status: 200 }));
    await expect(
      verifyTestRun(
        "https://api.example/",
        "https://app.example/",
        "project-key",
        "project/id",
        "run/id",
        fetch,
      ),
    ).resolves.toEqual({
      testRunId: "run/id",
      testRunUrl: "https://app.example/project/project%2Fid/test-runs/run%2Fid",
    });
    expect(fetch).toHaveBeenCalledWith(
      "https://api.example/v1/test-runs/run%2Fid",
      { headers: { CONFIDENT_API_KEY: "project-key" } },
    );
  });

  it("verifies with the key alone when the project is unknown", async () => {
    const fetch = vi
      .fn<typeof globalThis.fetch>()
      .mockResolvedValue(new Response("{}", { status: 200 }));
    await expect(
      verifyTestRun(
        "https://api.example",
        "https://app.example",
        "project-key",
        undefined,
        "run",
        fetch,
      ),
    ).resolves.toEqual({ testRunId: "run" });
  });

  it("rejects an unverified run", async () => {
    const fetch = vi
      .fn<typeof globalThis.fetch>()
      .mockResolvedValue(new Response("{}", { status: 404 }));
    await expect(
      verifyTestRun(
        "https://api.example",
        "https://app.example",
        "key",
        "project",
        "missing",
        fetch,
      ),
    ).rejects.toThrow("could not be verified");
  });

  it("reports the status the platform returned", async () => {
    const fetch = vi.fn<typeof globalThis.fetch>().mockResolvedValue(
      new Response(JSON.stringify({ data: { status: "IN_PROGRESS" } }), {
        status: 200,
      }),
    );
    await expect(
      verifyTestRun(
        "https://api.example",
        "https://app.example",
        "key",
        "project",
        "run",
        fetch,
      ),
    ).resolves.toMatchObject({ status: "IN_PROGRESS" });
  });

  it("rejects runs that errored or were cancelled", async () => {
    const fetch = vi.fn<typeof globalThis.fetch>().mockResolvedValue(
      new Response(JSON.stringify({ data: { status: "ERRORED" } }), {
        status: 200,
      }),
    );
    await expect(
      verifyTestRun(
        "https://api.example",
        "https://app.example",
        "key",
        "project",
        "run",
        fetch,
      ),
    ).rejects.toThrow("status ERRORED");
  });

  it("constructs the canonical project permalink", () => {
    expect(buildTestRunUrl("https://app.example", "project", "run")).toBe(
      "https://app.example/project/project/test-runs/run",
    );
  });
});
