import { describe, expect, it, vi } from "vitest";

import { ConfidentApi, type AuthSession } from "../src/api.js";

const session: AuthSession = {
  deviceCode: "d".repeat(32),
  userCode: "ABCD-EFGH",
  verificationUri: "https://app.example/auth/cli",
  verificationUriComplete: "https://app.example/auth/cli?code=ABCD",
  expiresIn: 10,
  interval: 1,
  protocolVersion: 1,
};

const jsonResponse = (body: unknown, status = 200): Response =>
  new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json" },
  });

describe("ConfidentApi auth polling", () => {
  it("binds source and project context when creating a session", async () => {
    const fetch = vi.fn<typeof globalThis.fetch>().mockResolvedValue(
      jsonResponse({
        success: true,
        data: { ...session, telemetryToken: "telemetry-token" },
      }),
    );
    const api = new ConfidentApi("https://api.example", { fetch });

    await expect(
      api.createAuthSession({
        source: "in_app_setup",
        organizationId: "org",
        projectId: "project",
      }),
    ).resolves.toMatchObject({ telemetryToken: "telemetry-token" });

    // The setup context is a strict contract: an extra key fails the request.
    expect(fetch).toHaveBeenCalledWith(
      "https://api.example/cli/auth/sessions",
      expect.objectContaining({
        body: JSON.stringify({
          context: {
            source: "in_app_setup",
            organizationId: "org",
            projectId: "project",
          },
        }),
      }),
    );
  });

  it("polls pending sessions until authenticated", async () => {
    let now = 0;
    const fetch = vi
      .fn<typeof globalThis.fetch>()
      .mockResolvedValueOnce(
        jsonResponse({ success: true, data: { status: "pending" } }),
      )
      .mockResolvedValueOnce(
        jsonResponse({
          success: true,
          data: {
            status: "authenticated",
            setupToken: "setup-token",
            email: "dev@example.com",
          },
        }),
      );
    const api = new ConfidentApi("https://api.example", {
      fetch,
      now: () => now,
      sleep: async (milliseconds) => {
        now += milliseconds;
      },
    });

    await expect(api.pollAuthSession(session)).resolves.toEqual({
      setupToken: "setup-token",
      email: "dev@example.com",
    });
    expect(fetch).toHaveBeenCalledTimes(2);
    expect(fetch.mock.calls[0]?.[1]?.body).toBe(
      JSON.stringify({ deviceCode: session.deviceCode }),
    );
  });

  it("surfaces denial responses without retrying", async () => {
    const fetch = vi
      .fn<typeof globalThis.fetch>()
      .mockResolvedValue(
        jsonResponse({ success: false, error: "Authorization denied." }, 403),
      );
    const api = new ConfidentApi("https://api.example", {
      fetch,
      now: () => 0,
      sleep: vi.fn(),
    });
    await expect(api.pollAuthSession(session)).rejects.toThrow(
      "Authorization denied.",
    );
    expect(fetch).toHaveBeenCalledOnce();
  });

  it("expires after the server-supplied lifetime", async () => {
    let now = 0;
    const api = new ConfidentApi("https://api.example", {
      fetch: vi
        .fn<typeof globalThis.fetch>()
        .mockResolvedValue(
          jsonResponse({ success: true, data: { status: "pending" } }),
        ),
      now: () => now,
      sleep: async (milliseconds) => {
        now += milliseconds;
      },
    });
    await expect(
      api.pollAuthSession({ ...session, expiresIn: 1 }),
    ).rejects.toThrow("expired");
  });
});
