// A leveled logger with Python's `logging` semantics, so `LOG_LEVEL`,
// `DEEPEVAL_LOG_STACK_TRACES` and the retry levels mean the same thing in both
// SDKs. Env vars are read per call, so changing them takes effect immediately.

import { envBool } from "@/env-flags";

export const LOG_LEVELS = {
  NOTSET: 0,
  DEBUG: 10,
  INFO: 20,
  WARNING: 30,
  WARN: 30,
  ERROR: 40,
  CRITICAL: 50,
  FATAL: 50,
} as const;

export type LogLevelName = keyof typeof LOG_LEVELS;

const DEFAULT_LEVEL = LOG_LEVELS.WARNING;
const DEFAULT_RETRY_AFTER_LEVEL = LOG_LEVELS.ERROR;

/** Accepts both names ("warning") and numbers ("30"), like Python. */
export function parseLogLevel(
  value: string | number | null | undefined,
): number | undefined {
  if (value === null || value === undefined || value === "") return undefined;
  if (typeof value === "number") {
    return Number.isFinite(value) ? value : undefined;
  }
  const text = value.trim();
  const named = LOG_LEVELS[text.toUpperCase() as LogLevelName];
  if (named !== undefined) return named;
  const numeric = Number(text);
  return Number.isFinite(numeric) ? numeric : undefined;
}

export function isValidLogLevel(value: string): boolean {
  return parseLogLevel(value) !== undefined;
}

function levelName(level: number): string {
  const match = Object.entries(LOG_LEVELS)
    .filter(([name]) => !["WARN", "FATAL"].includes(name))
    .reverse()
    .find(([, value]) => level >= value);
  return match ? match[0] : "NOTSET";
}

/** `NOTSET` means "inherit", which resolves to WARNING as in Python. */
export function getLogLevel(): number {
  const level = parseLogLevel(process.env.LOG_LEVEL);
  return level === undefined || level === LOG_LEVELS.NOTSET
    ? DEFAULT_LEVEL
    : level;
}

export function shouldLogStackTraces(): boolean {
  return envBool("DEEPEVAL_LOG_STACK_TRACES") ?? false;
}

function format(level: number, name: string, message: string): string {
  return `[${levelName(level)}] ${name}: ${message}`;
}

function render(args: unknown[]): string[] {
  return args.map((arg) => {
    if (arg instanceof Error) {
      return shouldLogStackTraces()
        ? (arg.stack ?? `${arg.name}: ${arg.message}`)
        : `${arg.name}: ${arg.message}`;
    }
    return typeof arg === "string" ? arg : JSON.stringify(arg);
  });
}

export interface Logger {
  log(level: number, message: string, ...args: unknown[]): void;
  debug(message: string, ...args: unknown[]): void;
  info(message: string, ...args: unknown[]): void;
  warning(message: string, ...args: unknown[]): void;
  error(message: string, ...args: unknown[]): void;
  critical(message: string, ...args: unknown[]): void;
  isEnabledFor(level: number): boolean;
}

export function getLogger(name: string): Logger {
  const emit = (level: number, message: string, args: unknown[]): void => {
    if (level < getLogLevel()) return;
    const line = format(level, name, message);
    const extra = render(args);
    const stream = level >= LOG_LEVELS.WARNING ? console.error : console.log;
    stream(line, ...extra);
  };

  return {
    log: (level, message, ...args) => emit(level, message, args),
    debug: (message, ...args) => emit(LOG_LEVELS.DEBUG, message, args),
    info: (message, ...args) => emit(LOG_LEVELS.INFO, message, args),
    warning: (message, ...args) => emit(LOG_LEVELS.WARNING, message, args),
    error: (message, ...args) => emit(LOG_LEVELS.ERROR, message, args),
    critical: (message, ...args) => emit(LOG_LEVELS.CRITICAL, message, args),
    isEnabledFor: (level) => level >= getLogLevel(),
  };
}

export function retryBeforeLevel(): number {
  return (
    parseLogLevel(process.env.DEEPEVAL_RETRY_BEFORE_LOG_LEVEL) ??
    parseLogLevel(process.env.LOG_LEVEL) ??
    LOG_LEVELS.INFO
  );
}

export function retryAfterLevel(): number {
  return (
    parseLogLevel(process.env.DEEPEVAL_RETRY_AFTER_LOG_LEVEL) ??
    DEFAULT_RETRY_AFTER_LEVEL
  );
}

/** Must run before any gRPC channel exists: grpc-js reads these at load time. */
export function applyGrpcLoggingEnv(): void {
  if (!(envBool("DEEPEVAL_GRPC_LOGGING") ?? false)) return;
  if (!process.env.GRPC_VERBOSITY) process.env.GRPC_VERBOSITY = "DEBUG";
  if (!process.env.GRPC_TRACE) process.env.GRPC_TRACE = "all";
}
