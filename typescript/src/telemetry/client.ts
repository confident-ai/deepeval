// Port of `deepeval/telemetry/client.py`. `PostHogBackend` is the only module
// allowed to import `posthog-node`, so swapping vendors is one new class.

import { getVersion } from "@/cli/version";
import { getSettings } from "@/config/settings";
import { getLogger } from "@/logger";
import { TELEMETRY_SCHEMA_VERSION, Event } from "@/telemetry/events";
import { getIdentity, isLoggedIn } from "@/telemetry/identity";
import {
  Language,
  mergedWith,
  type EventProperties,
  type PropValue,
} from "@/telemetry/properties";
import { detectRuntime } from "@/telemetry/runtime";

const POSTHOG_PROJECT_API_KEY =
  "phc_IXvGRcscJJoIb049PtjIZ65JnXQguOUZ5B5MncunFdB";
const POSTHOG_HOST = "https://us.i.posthog.com";

const logger = getLogger("telemetry");

export function telemetryOptOut(): boolean {
  return Boolean(getSettings().DEEPEVAL_TELEMETRY_OPT_OUT);
}

export interface TelemetryBackend {
  capture(
    anonymousId: string,
    event: Event,
    properties: Record<string, PropValue>,
  ): void;
  flush(): void;
}

/** Used when telemetry is opted out, so call sites need no branching. */
export class NoopBackend implements TelemetryBackend {
  capture(): void {}
  flush(): void {}
}

/**
 * The import is deferred to construction so an opted-out process never loads
 * the client at all.
 */
export class PostHogBackend implements TelemetryBackend {
  private readonly client: {
    capture(payload: {
      distinctId: string;
      event: string;
      properties: Record<string, PropValue>;
    }): void;
    flush(): Promise<void> | void;
  };

  constructor() {
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    const posthog = require("posthog-node") as typeof import("posthog-node");
    const { PostHog } = posthog;
    this.client = new PostHog(POSTHOG_PROJECT_API_KEY, { host: POSTHOG_HOST });
  }

  capture(
    anonymousId: string,
    event: Event,
    properties: Record<string, PropValue>,
  ): void {
    this.client.capture({ distinctId: anonymousId, event, properties });
  }

  flush(): void {
    // A flush that fails or outlives the process must not surface as an
    // unhandled rejection in the user's application.
    void Promise.resolve(this.client.flush()).catch(() => {});
  }
}

let backend: TelemetryBackend | null = null;

export function getBackend(): TelemetryBackend {
  if (backend === null) {
    if (telemetryOptOut()) {
      backend = new NoopBackend();
    } else {
      try {
        backend = new PostHogBackend();
      } catch (error) {
        logger.debug("Telemetry backend unavailable", error);
        backend = new NoopBackend();
      }
    }
  }
  return backend;
}

export function setBackend(next: TelemetryBackend | null): void {
  backend = next;
}

const installed = new Set<string>();

/** Returns true the first time each integration is seen. */
export function registerIntegration(name: string): boolean {
  if (installed.has(name)) return false;
  installed.add(name);
  return true;
}

export function installedIntegrations(): string[] {
  return [...installed].sort();
}

/** Stamped on every event. */
export function baseProperties(): EventProperties {
  const identity = getIdentity();
  const active = installedIntegrations();
  return {
    schemaVersion: TELEMETRY_SCHEMA_VERSION,
    sdkLanguage: Language.TYPESCRIPT,
    sdkVersion: getVersion(),
    runtime: detectRuntime(),
    userStatus: identity.status,
    userId: identity.anonymousId,
    loggedIn: isLoggedIn(),
    integrations: active,
    integrationsCount: active.length,
    integrationsPrimary: active[0] ?? "none",
  };
}

export function capture(event: Event, properties: EventProperties): void {
  if (telemetryOptOut()) return;
  try {
    const payload = mergedWith(baseProperties(), properties);
    getBackend().capture(getIdentity().anonymousId, event, payload);
  } catch (error) {
    // Telemetry must never break a user's evaluation.
    logger.debug(`Failed to capture ${event}`, error);
  }
}

export function flush(): void {
  try {
    getBackend().flush();
  } catch (error) {
    logger.debug("Failed to flush telemetry", error);
  }
}
