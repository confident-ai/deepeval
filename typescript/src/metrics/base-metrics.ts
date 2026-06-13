import { DeepEvalBaseLLM } from "../models";
import { SingleTurnParams } from "../test-case";

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

/**
 * Shared state + behavior for every metric (single-turn and conversational).
 * Holds the result fields, the progress indicator, and cost accrual; the
 * subclasses ({@link BaseMetric}, {@link BaseConversationalMetric}) add the
 * `requiredParams` type and the `measure(testCase)` signature.
 */
export abstract class BaseMetricCore {
  threshold: number;
  score?: number;
  scoreBreakdown?: Record<string, any>;
  reason?: string;
  success?: boolean;
  evaluationModel?: string;
  strictMode: boolean = false;
  verboseMode: boolean = false;
  includeReason: boolean = false;
  showIndicator: boolean = true;
  error?: string;
  evaluationCost?: number;
  verboseLogs?: string;
  skipped: boolean = false;
  requiresTrace: boolean = false;
  model?: DeepEvalBaseLLM;
  usingNativeModel?: boolean = undefined;
  private _spinner: import("ora").Ora | null = null;
  private _barTimer: ReturnType<typeof setInterval> | null = null;

  constructor(
    threshold: number,
    options?: {
      strictMode?: boolean;
      verboseMode?: boolean;
      includeReason?: boolean;
      showIndicator?: boolean;
    },
  ) {
    this.threshold = threshold;
    if (options) {
      this.strictMode = options.strictMode ?? this.strictMode;
      this.verboseMode = options.verboseMode ?? this.verboseMode;
      this.includeReason = options.includeReason ?? this.includeReason;
      this.showIndicator = options.showIndicator ?? this.showIndicator;
    }
  }

  abstract isSuccessful(): boolean;

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

  accrueCost(cost: number | null): void {
    if (cost != null && this.evaluationCost != null) {
      this.evaluationCost += cost;
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
