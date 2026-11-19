/**
 * Clack ships a different color per message kind (blue info, green success and
 * step, yellow warn, red error, magenta spinner). This narrows all of them to
 * the accent, leaving ember for the two kinds the user must act on.
 */

import {
  log as clackLog,
  spinner as clackSpinner,
  S_ERROR,
  S_INFO,
  S_STEP_SUBMIT,
  S_SUCCESS,
  S_WARN,
  type SpinnerResult,
} from "@clack/prompts";

import { accent, alert } from "./theme.js";

type Message = string | string[];

export const log = {
  message: (message: Message): void => clackLog.message(message),
  info: (message: Message): void =>
    clackLog.message(message, { symbol: accent(S_INFO) }),
  step: (message: Message): void =>
    clackLog.message(message, { symbol: accent(S_STEP_SUBMIT) }),
  success: (message: Message): void =>
    clackLog.message(message, { symbol: accent(S_SUCCESS) }),
  warn: (message: Message): void =>
    clackLog.message(message, { symbol: alert(S_WARN) }),
  error: (message: Message): void =>
    clackLog.message(message, { symbol: alert(S_ERROR) }),
};

/**
 * Clack ends a spinner with a green (or red) symbol it does not let us style,
 * so the final line is erased and reprinted through the themed log instead.
 */
export const spinner = (): SpinnerResult => {
  const inner = clackSpinner({ styleFrame: accent });
  const finish =
    (write: (message: string) => void) =>
    (message = ""): void => {
      inner.clear();
      if (message) write(message);
    };
  return {
    start: inner.start,
    message: inner.message,
    clear: inner.clear,
    stop: finish(log.step),
    cancel: finish(log.warn),
    error: finish(log.error),
    get isCancelled() {
      return inner.isCancelled;
    },
  };
};
