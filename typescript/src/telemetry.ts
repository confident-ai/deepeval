import crypto from "crypto";
import net from "net";

import { trace, Span, SpanKind, SpanStatusCode } from "@opentelemetry/api";
import { BatchSpanProcessor } from "@opentelemetry/sdk-trace-base";
import { NodeTracerProvider } from "@opentelemetry/sdk-trace-node";
import { OTLPTraceExporter } from "@opentelemetry/exporter-trace-otlp-grpc";
import { PostHog } from "posthog-node";

import Sentry from "@sentry/node";

export enum Feature {
  REDTEAMING = "redteaming",
  SYNTHESIZER = "synthesizer",
  EVALUATION = "evaluation",
  COMPONENT_EVALUATION = "component_evaluation",
  GUARDRAIL = "guardrail",
  BENCHMARK = "benchmark",
  CONVERSATION_SIMULATOR = "conversation_simulator",
  UNKNOWN = "unknown",
  TRACING_INTEGRATION = "tracing_integration",
}

import * as fs from "fs";
import * as path from "path";

const HIDDEN_DIR = ".deepeval";
const KEY_FILE = "deepeval.key";
const TELEMETRY_DATA_FILE_TS = ".deepeval_telemetry.txt";
const TELEMETRY_PATH = path.join(HIDDEN_DIR, TELEMETRY_DATA_FILE_TS);

// -------------------
// Config
// -------------------
const NEW_RELIC_LICENSE_KEY = "1711c684db8a30361a7edb0d0398772cFFFFNRAL";
const NEW_RELIC_OTLP_ENDPOINT = "https://otlp.nr-data.net:4317";
const POSTHOG_API_KEY = "phc_IXvGRcscJJoIb049PtjIZ65JnXQguOUZ5B5MncunFdB";
const POSTHOG_HOST = "https://us.i.posthog.com";

// -------------------
// Telemetry Helpers
// -------------------
function telemetryOptOut(): boolean {
  return process.env.DEEPEVAL_TELEMETRY_OPT_OUT === "1";
}

function _blockedByFirewall(): boolean {
  try {
    const socket = new net.Socket();
    socket.setTimeout(3000);
    socket.connect(80, "www.google.com");
    socket.destroy();
    return false;
  } catch (_error) {
    return true;
  }
}

async function getAnonymousPublicIp() {
  try {
    const response = await fetch("https://api.ipify.org?format=json");
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    const data = await response.json();
    return data.ip;
  } catch (error) {
    console.error("Error getting anonymous public IP:", error);
    return null;
  }
}

// -------------------
// Initialize Sentry, PostHog, New Relic
// -------------------

let anonymousPublicIp: string | null = null;
let isInitialized = false;
let posthog: PostHog | null = null;
let tracer: ReturnType<typeof trace.getTracer> | null = null;

// -------------------
// Context Manager Equivalent
// -------------------

function initializeTelemetry() {
  if (isInitialized || telemetryOptOut()) {
    return;
  }

  Sentry.init({
    dsn: "https://5ef587d58109ee45d6544f3657efdd1f@o4506098477236224.ingest.sentry.io/4506098479136768",
    tracesSampleRate: 1.0,
    sendDefaultPii: false,
    attachStacktrace: false,
    defaultIntegrations: false,
  });

  const otlpExporter = new OTLPTraceExporter({
    url: NEW_RELIC_OTLP_ENDPOINT,
    headers: {
      "api-key": NEW_RELIC_LICENSE_KEY,
    },
  });

  const spanProcessor = new BatchSpanProcessor(otlpExporter);
  const nodeProvider = new NodeTracerProvider({
    spanProcessors: [spanProcessor],
  });

  trace.setGlobalTracerProvider(nodeProvider);
  tracer = trace.getTracer(__filename);

  posthog = new PostHog(POSTHOG_API_KEY, {
    host: POSTHOG_HOST,
    flushAt: 1,
    flushInterval: 0,
  });

  isInitialized = true;
}

export async function withCaptureTracingIntegration<T>(
  integrationName: string,
  callback: (span?: Span) => Promise<T> | T,
): Promise<T> {
  if (telemetryOptOut()) {
    return callback();
  }

  initializeTelemetry();

  const event = `Tracing Integration: deepeval-ts.integrations.${integrationName}`;
  const distinctId = await getUniqueId();

  if (anonymousPublicIp === null) {
    anonymousPublicIp = await getAnonymousPublicIp();
  }

  const properties: Record<string, any> = {
    logged_in_with: await getLoggedInWith(),
    environment: "other",
    "user.status": await getStatus(),
    "user.unique_id": distinctId,
    "user.public_ip": anonymousPublicIp ?? "Unknown",
    "feature_status.tracing_integration": await getFeatureStatus({
      value: Feature.TRACING_INTEGRATION,
    }),
  };
  setLastFeature(Feature.TRACING_INTEGRATION);

  // PostHog Event
  if (!posthog) {
    console.error(`[telemetry] PostHog instance is null`);
    return callback();
  }
  try {
    posthog.capture({
      distinctId,
      event,
      properties,
    });
  } catch (err) {
    console.error("[telemetry] Failed to capture PostHog event:", err);
  }

  // New Relic / OTEL Span
  const span = tracer?.startSpan(event, { kind: SpanKind.INTERNAL });
  if (span) {
    for (const [key, value] of Object.entries(properties)) {
      span.setAttribute(key, value as any);
    }
    span.setAttribute("integration.name", integrationName);

    try {
      const result = await callback(span);
      span.setStatus({ code: SpanStatusCode.OK });
      return result;
    } catch (err) {
      span.setStatus({ code: SpanStatusCode.ERROR });
      span.recordException(err as Error);
      throw err;
    } finally {
      span.end();
      await posthog.shutdown();
    }
  } else {
    return callback();
  }
}

// -------------------
// Helper Functions
// -------------------

async function writeTelemetryFile(data: Record<string, string>): Promise<void> {
  if (telemetryOptOut()) {
    return;
  }

  const TELEMETRY_DIR = __dirname;
  const TELEMETRY_PATH = path.join(TELEMETRY_DIR, "telemetry.txt");
  if (typeof fs !== "undefined" && fs.promises && fs.promises.mkdir) {
    try {
      await fs.promises.mkdir(TELEMETRY_DIR, { recursive: true });
    } catch (error) {
      // Ignore if directory already exists or cannot be created
      console.error(error);
    }
  }

  const lines = Object.entries(data).map(([key, value]) => `${key}=${value}\n`);
  await fs.promises.writeFile(TELEMETRY_PATH, lines.join(""), "utf-8");
}

async function readTelemetryFile(): Promise<Record<string, string>> {
  const TELEMETRY_PATH = path.join(__dirname, "telemetry.txt");
  if (!fs.existsSync(TELEMETRY_PATH)) {
    return {};
  }

  const data: Record<string, string> = {};

  const fileContent = await fs.promises.readFile(TELEMETRY_PATH, "utf-8");
  const lines = fileContent.split("\n");
  for (const line of lines) {
    const trimmed = line.trim();
    if (trimmed === "") {
      continue;
    }

    const [key, value] = trimmed.split("=", 2);
    data[key] = value ?? "";
  }

  return data;
}

async function getLoggedInWith(): Promise<string> {
  const data = await readTelemetryFile();
  return data["LOGGED_IN_WITH"] ?? "NA";
}

async function getStatus(): Promise<string> {
  const data = await readTelemetryFile();
  return data["DEEPEVAL_STATUS"] ?? "new";
}

async function _getLastFeature(): Promise<string | undefined> {
  const data = await readTelemetryFile();
  const lastFeature = data["DEEPEVAL_LAST_FEATURE"];
  if (lastFeature) {
    return lastFeature;
  }
  return undefined;
}

async function setLastFeature(feature: Feature): Promise<void> {
  if (!feature || typeof feature !== "string") {
    throw new Error(`Invalid feature: ${feature}`);
  }

  const data = await readTelemetryFile();
  data["DEEPEVAL_LAST_FEATURE"] = feature;
  const featureStatusKey = `DEEPEVAL_${feature.toUpperCase()}_STATUS`;
  data[featureStatusKey] = "old";
  await writeTelemetryFile(data);
}

async function getFeatureStatus(feature: { value: string }): Promise<string> {
  const data = await readTelemetryFile();
  const featureStatusKey = `DEEPEVAL_${feature.value.toUpperCase()}_STATUS`;
  return data[featureStatusKey] ?? "new";
}

async function getUniqueId(): Promise<string> {
  if (telemetryOptOut()) {
    return "telemetry-opted-out";
  }

  const data = await readTelemetryFile();

  let uniqueId = data["DEEPEVAL_ID"];
  if (!uniqueId) {
    uniqueId = crypto.randomUUID();

    data["DEEPEVAL_ID"] = uniqueId;
    data["DEEPEVAL_STATUS"] = "new";
  } else {
    data["DEEPEVAL_STATUS"] = "old";
  }

  await writeTelemetryFile(data);

  return uniqueId;
}

// -------------------
// Move Folders
// -------------------
function _moveTelemetryFilesIfNeeded() {
  if (telemetryOptOut()) {
    return;
  }

  if (
    (fs.existsSync(KEY_FILE) && !fs.existsSync(HIDDEN_DIR)) ||
    !fs.lstatSync(HIDDEN_DIR).isDirectory()
  ) {
    const tempDeepevalFileName = ".deepeval_temp";
    fs.renameSync(KEY_FILE, tempDeepevalFileName);
    fs.mkdirSync(HIDDEN_DIR, { recursive: true });
    fs.renameSync(tempDeepevalFileName, path.join(HIDDEN_DIR, KEY_FILE));
  }

  fs.mkdirSync(HIDDEN_DIR, { recursive: true });

  if (fs.existsSync(TELEMETRY_DATA_FILE_TS)) {
    fs.renameSync(TELEMETRY_DATA_FILE_TS, TELEMETRY_PATH);
  }

  const cacheFile = ".deepeval-cache.json";
  if (fs.existsSync(cacheFile)) {
    fs.renameSync(cacheFile, path.join(HIDDEN_DIR, cacheFile));
  }

  const tempTestRunDataFile = ".temp_test_run_data.json";
  if (fs.existsSync(tempTestRunDataFile)) {
    fs.renameSync(
      tempTestRunDataFile,
      path.join(HIDDEN_DIR, tempTestRunDataFile),
    );
  }
}

// moveTelemetryFilesIfNeeded();
