import { runCommand, type CommandRunner } from "./git.js";

/** macOS bundle ids, the only shape worth passing to `osascript`. */
const BUNDLE_ID = /^[A-Za-z0-9.-]+$/;

export interface FocusOptions {
  platform?: NodeJS.Platform;
  /** Set by macOS to the app that owns this terminal, editors included. */
  bundleId?: string | undefined;
  isTty?: boolean;
  write?: (text: string) => void;
  runner?: CommandRunner;
}

/**
 * A browser cannot hand focus back, so the wizard asks for it once pairing
 * finishes. The bell is all that portably marks the tab; macOS can also raise
 * the owning app, which is the editor when this runs in its terminal.
 */
export const restoreTerminalFocus = async ({
  platform = process.platform,
  bundleId = process.env.__CFBundleIdentifier,
  isTty = Boolean(process.stdout.isTTY),
  write = (text) => void process.stdout.write(text),
  runner = runCommand,
}: FocusOptions = {}): Promise<void> => {
  if (isTty) write("\x07");
  if (platform !== "darwin" || !bundleId || !BUNDLE_ID.test(bundleId)) return;
  await runner(
    "osascript",
    ["-e", `tell application id "${bundleId}" to activate`],
    { cwd: process.cwd() },
  ).catch(() => undefined);
};
