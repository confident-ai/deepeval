import { describe, expect, it } from "vitest";

import {
  applyCustomSelection,
  enforceExclusiveSelection,
  selectProject,
  validateOrganization,
} from "../src/onboarding.js";

const onboarding = {
  state: "existing_user" as const,
  user: { email: "dev@example.com", name: "Dev" },
  organization: { id: "org-1", name: "Org" },
  projects: [
    { id: "allowed", name: "Allowed", canCreateApiKey: true },
    { id: "denied", name: "Denied", canCreateApiKey: false },
  ],
};

describe("onboarding helpers", () => {
  it("validates organization and project access", () => {
    expect(() => validateOrganization(onboarding, "org-1")).not.toThrow();
    expect(() => validateOrganization(onboarding, "other")).toThrow();
    expect(selectProject(onboarding, "allowed")).toEqual({
      id: "allowed",
      name: "Allowed",
    });
    expect(() => selectProject(onboarding, "denied")).toThrow("permission");
    expect(() => selectProject(onboarding, "missing")).toThrow("not available");
  });

  it("only skips the project prompt when a single project is manageable", () => {
    expect(selectProject(onboarding)).toEqual({
      id: "allowed",
      name: "Allowed",
    });
    expect(
      selectProject({
        ...onboarding,
        projects: [
          { id: "one", name: "One", canCreateApiKey: true },
          { id: "two", name: "Two", canCreateApiKey: true },
        ],
      }),
    ).toBeUndefined();
    expect(
      selectProject({
        ...onboarding,
        projects: [{ id: "denied", name: "Denied", canCreateApiKey: false }],
      }),
    ).toBeUndefined();
  });

  it("handles custom and exclusive questionnaire answers", () => {
    expect(applyCustomSelection(["A", "CUSTOM"], "CUSTOM", "Other")).toEqual([
      "A",
      "Other",
    ]);
    expect(enforceExclusiveSelection(["TOOLS", "NONE"], ["NONE"])).toEqual([
      "NONE",
    ]);
  });
});
