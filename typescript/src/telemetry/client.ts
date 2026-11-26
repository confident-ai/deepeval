// Port of `deepeval/telemetry/client.py`. Events go as plain JSON to deepeval's
// own telemetry host, and `HttpBackend` is the only place that knows the host.

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

const TELEMETRY_URL = "https://telemetry.deepeval.com/v1/sdk";

// The server accepts at most 100 events per request.
export const MAX_BATCH_SIZE = 100;
// Events beyond this are dropped rather than growing without bound offline.
export const MAX_QUEUE_SIZE = 1000;
const REQUEST_TIMEOUT_MS = 3000;
// Long enough for the events of one run to leave in a single request.
const SEND_DELAY_MS = 500;

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

interface QueuedEvent {
  event: string;
  anonymousId: string;
  properties: Record<string, PropValue>;
  timestamp: string;
}

/**
 * Posts events to deepeval's telemetry host. `capture` only queues; the queue
 * leaves shortly after, or at once on `flush`. Nothing here throws, rejects or
 * logs above debug level: a user never sees a telemetry failure.
 */
export class HttpBackend implements TelemetryBackend {
  private pending: QueuedEvent[] = [];
  private timer: ReturnType<typeof setTimeout> | null = null;

  constructor(private readonly url: string = TELEMETRY_URL) {}

  capture(
    anonymousId: string,
    event: Event,
    properties: Record<string, PropValue>,
  ): void {
    if (this.pending.length >= MAX_QUEUE_SIZE) return;
    this.pending.push({
      event,
      anonymousId,
      properties,
      timestamp: new Date().toISOString(),
    });
    if (this.timer === null) {
      this.timer = setTimeout(() => this.flush(), SEND_DELAY_MS);
      // A queued event must not keep the user's process alive.
      this.timer.unref?.();
    }
  }

  flush(): void {
    if (this.timer !== null) {
      clearTimeout(this.timer);
      this.timer = null;
    }
    while (this.pending.length > 0) {
      void this.send(this.pending.splice(0, MAX_BATCH_SIZE));
    }
  }

  private async send(events: QueuedEvent[]): Promise<void> {
    try {
      await fetch(this.url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ events }),
        signal: AbortSignal.timeout(REQUEST_TIMEOUT_MS),
      });
    } catch (error) {
      // Telemetry is best effort: a failed batch is dropped, not retried.
      logger.debug("Failed to send telemetry", error);
    }
  }
}

let backend: TelemetryBackend | null = null;

export function getBackend(): TelemetryBackend {
  if (backend === null) {
    if (telemetryOptOut()) {
      backend = new NoopBackend();
    } else {
      try {
        backend = new HttpBackend();
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
