import { LLMTestCase, SingleTurnParams, ToolCall } from "@/test-case";
import { resolveTemplate } from "@/templates";
import { generateWithSchema } from "@/metrics/utils";
import { G_EVAL_PARAMS } from "@/metrics/g-eval/utils";
import {
  BinaryJudgementVerdictSchema,
  MetricScoreReasonSchema,
  TaskNodeOutputSchema,
  nonBinaryVerdictSchema,
  type JudgementVerdict,
} from "@/metrics/dag/schema";
import {
  type AnyNode,
  type ChildMetric,
  type DagHostMetric,
  type ExecutableNode,
  type NodeContext,
  type VerdictLikeNode,
} from "@/metrics/dag/types";
import { cloneChildMetric, validateVerdictBranch } from "@/metrics/dag/utils";
import {
  judgementVerboseLog,
  taskVerboseLog,
  verdictVerboseLog,
} from "@/metrics/dag/verbose";

function formatValue(value: unknown): string {
  if (value == null) return "None";
  if (typeof value === "string") return value;
  if (value instanceof ToolCall) return JSON.stringify(value);
  if (Array.isArray(value)) {
    return (
      "[" +
      value
        .map((item) =>
          typeof item === "string" ? `'${item}'` : JSON.stringify(item),
        )
        .join(", ") +
      "]"
    );
  }
  return String(value);
}

abstract class BaseNode {
  readonly multiTurn = false;
  label?: string;
  evaluationParams?: SingleTurnParams[];

  validate(): void {}

  protected resolveText(ctx: NodeContext, sep = "\n\n"): string {
    let text = "";
    for (const parent of ctx.parents ?? []) {
      if (parent.nodeKind === "task") {
        const task = parent as TaskNode;
        text += `${task.outputLabel}:\n${ctx.outputs.get(parent)}${sep}`;
      }
    }
    const testCase = ctx.testCase as unknown as Record<string, unknown>;
    for (const param of this.evaluationParams ?? []) {
      text += `${G_EVAL_PARAMS[param] ?? param}:\n${formatValue(testCase[param])}\n`;
    }
    return text;
  }
}

export interface TaskNodeOptions {
  instructions: string;
  outputLabel: string;
  evaluationParams?: SingleTurnParams[];
  label?: string;
}

export class TaskNode extends BaseNode implements ExecutableNode {
  readonly nodeKind = "task" as const;
  readonly children: AnyNode[] = [];
  instructions: string;
  outputLabel: string;

  constructor(options: TaskNodeOptions) {
    super();
    this.instructions = options.instructions;
    this.outputLabel = options.outputLabel;
    this.evaluationParams = options.evaluationParams;
    this.label = options.label;
  }

  addNode<T extends TaskNode | BinaryJudgementNode | NonBinaryJudgementNode>(
    child: T,
  ): T {
    this.children.push(child);
    return child;
  }

  validate(): void {
    for (const child of this.children) {
      if (child.nodeKind === "verdict") {
        throw new Error(
          "A TaskNode must not have a VerdictNode as one of their 'children'.",
        );
      }
    }
  }

  async execute(ctx: NodeContext): Promise<unknown> {
    if (this.evaluationParams == null && ctx.parents == null) {
      throw new Error(
        "A TaskNode must have either a 'evaluationParams' or parent node(s).",
      );
    }
    const prompt = resolveTemplate(
      "metrics",
      "TaskNode",
      "generate_task_output",
      { instructions: this.instructions, text: this.resolveText(ctx) },
    );
    const { output } = await generateWithSchema(
      ctx.metric,
      prompt,
      TaskNodeOutputSchema,
    );
    return output;
  }

  verboseLog(depth: number, result: unknown): string {
    return taskVerboseLog(
      "______________________",
      "*******************************",
      "TaskNode",
      this.label,
      this.instructions,
      this.outputLabel,
      depth,
      result,
    );
  }
}

export interface JudgementNodeOptions {
  criteria: string;
  evaluationParams?: SingleTurnParams[];
  label?: string;
}

export interface AddVerdictOptions {
  score?: number;
  then?: AnyNode | ChildMetric;
}

export class BinaryJudgementNode extends BaseNode implements ExecutableNode {
  readonly nodeKind = "binaryJudgement" as const;
  readonly children: VerdictNode[] = [];
  criteria: string;

  constructor(options: JudgementNodeOptions) {
    super();
    this.criteria = options.criteria;
    this.evaluationParams = options.evaluationParams;
    this.label = options.label;
  }

  addVerdict(verdict: boolean, options: AddVerdictOptions = {}): VerdictNode {
    const node = new VerdictNode({
      verdict,
      score: options.score,
      child: options.then,
    });
    this.children.push(node);
    return node;
  }

  validate(): void {
    if (this.children.length !== 2) {
      throw new Error("BinaryJudgementNode must have exactly 2 children.");
    }
    const verdicts = this.children.map((child) => child.verdict);
    if (verdicts.some((verdict) => typeof verdict !== "boolean")) {
      throw new Error(
        "All children BinaryJudgementNode must have a boolean verdict.",
      );
    }
    if (
      verdicts.filter((verdict) => verdict === true).length !== 1 ||
      verdicts.filter((verdict) => verdict === false).length !== 1
    ) {
      throw new Error(
        "BinaryJudgementNode must have one True and one False VerdictNode child.",
      );
    }
  }

  async execute(ctx: NodeContext): Promise<JudgementVerdict> {
    const prompt = resolveTemplate(
      "metrics",
      "BinaryJudgement",
      "generate_binary_verdict",
      { criteria: this.criteria, text: this.resolveText(ctx) },
    );
    return generateWithSchema(ctx.metric, prompt, BinaryJudgementVerdictSchema);
  }

  verboseLog(depth: number, result: unknown): string {
    return judgementVerboseLog(
      "BinaryJudgementNode",
      34,
      48,
      this.label,
      this.criteria,
      depth,
      result as JudgementVerdict,
    );
  }
}

export class NonBinaryJudgementNode extends BaseNode implements ExecutableNode {
  readonly nodeKind = "nonBinaryJudgement" as const;
  readonly children: VerdictNode[] = [];
  criteria: string;
  private verdictOptions: string[] = [];

  constructor(options: JudgementNodeOptions) {
    super();
    this.criteria = options.criteria;
    this.evaluationParams = options.evaluationParams;
    this.label = options.label;
  }

  addVerdict(verdict: string, options: AddVerdictOptions = {}): VerdictNode {
    const node = new VerdictNode({
      verdict,
      score: options.score,
      child: options.then,
    });
    this.children.push(node);
    return node;
  }

  validate(): void {
    if (this.children.length === 0) {
      throw new Error("NonBinaryJudgementNode must have at least one child.");
    }
    const seen = new Set<string>();
    for (const child of this.children) {
      if (typeof child.verdict !== "string") {
        throw new Error(
          "The verdict attribute of all NonBinaryJudgementNode children must be a string.",
        );
      }
      if (seen.has(child.verdict)) {
        throw new Error(
          `Duplicate verdict found: ${child.verdict} in children of NonBinaryJudgementNode.`,
        );
      }
      seen.add(child.verdict);
    }
    this.verdictOptions = [...seen];
  }

  async execute(ctx: NodeContext): Promise<JudgementVerdict> {
    const prompt = resolveTemplate(
      "metrics",
      "NonBinaryJudgement",
      "generate_non_binary_verdict",
      {
        criteria: this.criteria,
        text: this.resolveText(ctx, "\n"),
        options: this.verdictOptions,
      },
    );
    return generateWithSchema(
      ctx.metric,
      prompt,
      nonBinaryVerdictSchema(this.verdictOptions),
    );
  }

  verboseLog(depth: number, result: unknown): string {
    return judgementVerboseLog(
      "NonBinaryJudgementNode",
      37,
      53,
      this.label,
      this.criteria,
      depth,
      result as JudgementVerdict,
    );
  }
}

export interface VerdictNodeOptions {
  verdict: string | boolean;
  score?: number;
  child?: AnyNode | ChildMetric;
}

export class VerdictNode extends BaseNode implements VerdictLikeNode {
  readonly nodeKind = "verdict" as const;
  readonly verdict: string | boolean;
  readonly score?: number;
  readonly child?: AnyNode | ChildMetric;

  constructor(options: VerdictNodeOptions) {
    super();
    validateVerdictBranch(options.score, options.child, "VerdictNode");
    this.verdict = options.verdict;
    this.score = options.score;
    this.child = options.child;
  }

  async generateReason(metric: DagHostMetric): Promise<string> {
    const prompt = resolveTemplate(
      "metrics",
      "VerdictNode",
      "generate_reason",
      {
        verbose_steps: metric.verboseSteps,
        score: metric.score,
        name: metric.name,
      },
    );
    const { reason } = await generateWithSchema(
      metric,
      prompt,
      MetricScoreReasonSchema,
    );
    return reason;
  }

  async runChildMetric(
    metric: DagHostMetric,
    testCase: LLMTestCase,
  ): Promise<ChildMetric> {
    const copy = cloneChildMetric(this.child as ChildMetric, metric);
    await copy.measure(testCase);
    return copy;
  }

  verboseLog(depth: number, childMetric?: ChildMetric): string {
    return verdictVerboseLog(
      "________________________",
      "**********************************",
      "VerdictNode",
      this.verdict,
      depth,
      childMetric,
    );
  }
}
