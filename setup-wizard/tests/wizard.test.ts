import { describe, expect, it } from "vitest";

import { isWizardCancellation, pairingUrlForApp } from "../src/wizard.js";

describe("wizard helpers", () => {
  it("moves pairing paths to the configured app origin", () => {
    expect(
      pairingUrlForApp(
        "https://self-hosted.example/base",
        "https://default.example/auth/cli?code=ABCD&region=US",
      ),
    ).toBe("https://self-hosted.example/auth/cli?code=ABCD&region=US");
  });

  it("treats only its own cancellation as a cancellation", () => {
    expect(isWizardCancellation(new Error("Setup failed."))).toBe(false);
  });
});
