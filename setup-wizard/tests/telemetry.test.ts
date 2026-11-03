import { describe, expect, it, vi } from "vitest";

import {
  classifyErrorCode,
  redactTelemetry,
  SetupTelemetry,
} from "../src/telemetry.js";

describe("telemetry redaction", () => {
  it("redacts sensitive keys and token-shaped values recursively", () => {
    expect(
      redactTelemetry({
        mode: "built-in",
        apiKey: "secret",
        nested: {
          authorization: "Bearer token",
          detail: "proj_abcdefghijklmnopqrstuvwxyz1234",
        },
      }),
    ).toEqual({
      mode: "built-in",
      apiKey: "[REDACTED]",
      nested: {
        authorization: "[REDACTED]",
        detail: "[REDACTED]",
      },
    });
  });

  it("classifies failures into codes that carry no message text", () => {
    expect(classifyErrorCode(new Error("claude timed out after 1800s."))).toBe(
      "timeout",
    );
    expect(classifyErrorCode(new Error("EACCES: open .env.local"))).toBe(
      "permission_denied",
    );
    expect(classifyErrorCode(new Error("fetch failed"))).toBe("network");
    expect(classifyErrorCode(new Error("codex exited with 1"))).toBe(
      "command_failed",
    );
    expect(classifyErrorCode(new Error("Project abc is not available."))).toBe(
      "invalid_configuration",
    );
    expect(classifyErrorCode("something else entirely")).toBe("unknown");
  });

  it("does not send when disabled", async () => {
    const fetch = vi.fn<typeof globalThis.fetch>();
    const telemetry = new SetupTelemetry("https://api.example", {
      fetch,
      disabled: true,
    });
    await telemetry.send({ event: "setup_started" });
    expect(fetch).not.toHaveBeenCalled();
  });

  it("never fails setup when posting fails", async () => {
    const telemetry = new SetupTelemetry("https://api.example", {
      fetch: vi
        .fn<typeof globalThis.fetch>()
        .mockRejectedValue(new Error("offline")),
      disabled: false,
    });
    telemetry.setTelemetryToken("telemetry-token");
    await expect(
      telemetry.send({ event: "setup_completed" }),
    ).resolves.toBeUndefined();
  });

  it("authenticates event delivery with the pairing telemetry token", async () => {
    const fetch = vi
      .fn<typeof globalThis.fetch>()
      .mockResolvedValue(new Response(null, { status: 202 }));
    const telemetry = new SetupTelemetry("https://api.example", {
      fetch,
      disabled: false,
    });
    telemetry.setTelemetryToken("telemetry-token");

    await telemetry.send({
      event: "setup_completed",
      step: "evaluation",
      result: "succeeded",
    });

    expect(fetch).toHaveBeenCalledWith(
      "https://api.example/cli/analytics",
      expect.objectContaining({
        headers: expect.objectContaining({
          authorization: "Bearer telemetry-token",
        }),
      }),
    );
  });
});
