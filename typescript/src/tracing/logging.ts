import { getSettings } from "../config/settings";
import { CONFIDENT_TRACE_VERBOSE } from "../constants";
import { Environment } from "./utils";

export enum TraceWorkerStatus {
  SUCCESS = "success",
  FAILURE = "failure",
  WARNING = "warning",
}

export const statusColorMarkup = {
  [TraceWorkerStatus.SUCCESS]: (text: string) => `\x1b[32m${text}\x1b[0m`, // Green
  [TraceWorkerStatus.FAILURE]: (text: string) => `\x1b[31m${text}\x1b[0m`, // Red
  [TraceWorkerStatus.WARNING]: (text: string) => `\x1b[33m${text}\x1b[0m`, // Yellow
};

export function printTraceStatus(
  traceWorkerStatus: TraceWorkerStatus,
  message: string,
  description?: string,
  environment?: Environment,
  evaluating?: boolean,
): void {
  const settings = getSettings();
  const tracingVerbose =
    settings.CONFIDENT_TRACE_VERBOSE === undefined
      ? true
      : !!settings.CONFIDENT_TRACE_VERBOSE;

  if (!tracingVerbose || evaluating) {
    return;
  }

  const messagePrefix = "\x1b[2m[Confident AI Trace Log]\x1b[0m";

  let coloredMessage = message;
  if (traceWorkerStatus === TraceWorkerStatus.SUCCESS) {
    coloredMessage = statusColorMarkup[TraceWorkerStatus.SUCCESS](message);
  } else if (traceWorkerStatus === TraceWorkerStatus.FAILURE) {
    coloredMessage = statusColorMarkup[TraceWorkerStatus.FAILURE](message);
  } else if (traceWorkerStatus === TraceWorkerStatus.WARNING) {
    coloredMessage = statusColorMarkup[TraceWorkerStatus.WARNING](message);
  }

  const envText = environment ? `[${environment}]` : "";

  if (description) {
    console.log(
      messagePrefix,
      envText,
      coloredMessage + ":",
      description,
      `\nTo disable dev logging, set ${CONFIDENT_TRACE_VERBOSE}=0 as an environment variable.`,
    );
  } else {
    console.log(messagePrefix, envText, coloredMessage);
  }
}
