import { isVerboseMode } from "@/env-flags";
import { DeepEvalBaseLLM } from "@/models";
import { SingleTurnParams } from "@/test-case";
import { observeMethods } from "@/tracing/internal";
import { resolveTemplate } from "@/templates";
import {
  camelizeVars,
  decamelizeVars,
  findOverride,
  type TemplateVars,
} from "@/templates/override";

// An indeterminate "pulse" progress bar (a bright window sliding across a dim
// track, wrapping), mirroring rich's animated BarColumn in Python's per-metric
// indicator. Re-rendered each animation frame.
function animatedBar(frame: number, width = 24, window = 6): string {
  const PURPLE = "\x1b[38;2;106;0;255m";
  const DIM = "\x1b[38;2;70;70;82m";
  const RESET = "\x1b[0m";
  const pos = frame % width;
  let bar = "";
  for (let i = 0; i < width; i++) {
    const lit = (i - pos + width) % width < window;
    bar += lit ? `${PURPLE}━` : `${DIM}━`;
  }
  return bar + RESET;
}

/** `undefined` takes the default; an explicit `null` (score-only) must survive. */
export function resolveThreshold(
  threshold: number | null | undefined,
  fallback: number,
): number | null {
  return threshold === undefined ? fallback : threshold;
}

/**
 * Shared state + behavior for every metric (single-turn and conversational).
 * Holds the result fields, the progress indicator, and cost accrual; the
 * subclasses ({@link BaseMetric}, {@link BaseConversationalMetric}) add the
 * `requiredParams` type and the `measure(testCase)` signature.
 */
export abstract class BaseMetricCore {
  /** `null` = score-only: scores, but gives no verdict. */
  threshold: number | null;
  score?: number;
  scoreBreakdown?: Record<string, any>;
  reason?: string;
  success?: boolean;
  evaluationModel?: string;
  strictMode: boolean = false;
  verboseMode: boolean = false;
  includeReason: boolean = false;
  showIndicator: boolean = true;
  /** Scores, but never fails a test case. */
  flaky: boolean = false;
  error?: string;
  evaluationCost?: number;
  verboseLogs?: string;
  skipped: boolean = false;
  requiresTrace: boolean = false;
  model?: DeepEvalBaseLLM;
  usingNativeModel?: boolean = undefined;
  /** Direction of the threshold comparison; the safety metrics flip it. */
  protected higherIsBetter: boolean = true;
  /** Set from `testCase.multimodal` by the param-check helpers. */
  multimodal: boolean = false;
  /** Opt-in: only these render their `{% if multimodal %}` branches, as in Python. */
  protected multimodalAware: boolean = false;
  /** This metric's key in the shared template bundle. */
  protected templateClass: string = "";
  protected evaluationTemplate?: unknown;
  private _spinner: import("ora").Ora | null = null;
  private _barTimer: ReturnType<typeof setInterval> | null = null;

  constructor(
    threshold: number | null,
    options?: {
      strictMode?: boolean;
      verboseMode?: boolean;
      includeReason?: boolean;
      showIndicator?: boolean;
      flaky?: boolean;
      evaluationTemplate?: unknown;
    },
  ) {
    observeMethods(this, { methods: ["measure"] });
    this.threshold = threshold;
    // `--verbose` raises the floor for metrics that didn't opt out explicitly.
    this.verboseMode = isVerboseMode();
    if (options) {
      this.strictMode = options.strictMode ?? this.strictMode;
      this.verboseMode = options.verboseMode ?? this.verboseMode;
      this.includeReason = options.includeReason ?? this.includeReason;
      this.showIndicator = options.showIndicator ?? this.showIndicator;
      this.flaky = options.flaky ?? this.flaky;
      this.evaluationTemplate = options.evaluationTemplate;
    }
  }

  /**
   * An explicit `templateClass` borrows another metric's template and skips the
   * override, so an override can't hijack a prompt that isn't this metric's own.
   */
  protected getPrompt(
    method: string,
    vars: Record<string, unknown> = {},
    opts: { templateClass?: string; strict?: boolean } = {},
  ): string {
    const { templateClass, strict = true } = opts;
    // An explicit `multimodal` in `vars` still wins.
    const resolved: Record<string, unknown> = {
      multimodal: this.multimodalAware && this.multimodal,
      ...vars,
    };
    const render = (v: Record<string, unknown>) =>
      resolveTemplate(
        "metrics",
        templateClass ?? this.templateClass,
        method,
        v,
        { strict },
      );

    if (templateClass === undefined) {
      const override = findOverride(this.evaluationTemplate, method);
      if (override) {
        const keys = Object.keys(resolved);
        const context = camelizeVars(resolved) as TemplateVars;
        return override(context, (overridden) =>
          render(decamelizeVars(overridden ?? context, keys)),
        );
      }
    }
    return render(resolved);
  }

  /**
   * Threshold is checked before the error, as in Python: a metric with nothing to
   * decide has no verdict to give, even when it failed.
   */
  isSuccessful(): boolean | undefined {
    if (this.threshold === null) {
      this.success = undefined;
    } else if (this.error != null || this.score == null) {
      this.success = false;
    } else {
      this.success = this.higherIsBetter
        ? this.score >= this.threshold
        : this.score <= this.threshold;
    }
    return this.success;
  }

  /** Strict mode collapses a near-miss score to the failing extreme. */
  protected applyStrictMode(score: number): number {
    if (!this.strictMode || this.threshold === null) return score;
    if (this.higherIsBetter) return score < this.threshold ? 0 : score;
    return score > this.threshold ? 1 : score;
  }

  /** The "✨ You're running DeepEval's latest …" description line (mirrors Python). */
  describe(): string {
    const PURPLE = "\x1b[38;2;106;0;255m";
    const GRAY = "\x1b[38;2;55;65;81m";
    const RESET = "\x1b[0m";
    const strict = this.strictMode ? "True" : "False";
    const asyncMode = "True";
    // Deterministic metrics have no model — omit the "using <model>" clause.
    const usingClause = this.evaluationModel
      ? `using ${this.evaluationModel}, `
      : "";
    return (
      `✨ You're running DeepEval's latest ${PURPLE}${this.name} Metric${RESET}! ` +
      `${GRAY}(${usingClause}strict=${strict}, async_mode=${asyncMode})...${RESET}`
    );
  }

  protected async startProgress(): Promise<void> {
    if (!this.showIndicator) return;
    const ora = (await import("ora")).default;
    const messageTail = this.describe();

    const spinner = ora({
      text: `${animatedBar(0)} ${messageTail}`,
      color: "magenta",
      stream: process.stderr,
    }).start();
    this._spinner = spinner;
    // Animate the bar like Python's rich progress (an indeterminate pulse).
    let frame = 0;
    this._barTimer = setInterval(() => {
      frame += 1;
      spinner.text = `${animatedBar(frame)} ${messageTail}`;
    }, 90);
  }

  protected stopProgress(): void {
    if (this._barTimer) {
      clearInterval(this._barTimer);
      this._barTimer = null;
    }
    this._spinner?.stop();
    this._spinner = null;
  }

  /**
   * A cost we can't price makes the running total meaningless, so it becomes
   * `undefined` ("unknown") rather than staying at its current value, which
   * would read as free. Sticky, as in Python's `_accrue_cost`.
   */
  accrueCost(cost: number | null): void {
    if (cost != null && this.evaluationCost != null) {
      this.evaluationCost += cost;
    } else {
      this.evaluationCost = undefined;
    }
  }

  get name(): string {
    return "Base Metric";
  }
}

export abstract class BaseMetric extends BaseMetricCore {
  protected requiredParams: Array<SingleTurnParams> = [];

  abstract measure(testCase: any, ...args: any[]): number | Promise<number>;
}
