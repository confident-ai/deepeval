import { ConversationalTestCase, MultiTurnParams, Turn } from "@/test-case";
import { resolveTemplate } from "@/templates";
import { generateWithSchema } from "@/metrics/utils";
import { CONVERSATIONAL_G_EVAL_PARAMS } from "@/metrics/conversational-g-eval/utils";
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

export type TurnWindow = [number, number];

export function isValidTurnWindow(
  turnWindow: TurnWindow,
  turns: Turn[],
): boolean {
  if (turnWindow.length !== 2) {
    throw new Error(
      "A 'turnWindow' must have only 2 indices representing start and end",
    );
  }
  const [start, end] = turnWindow;
  if (
    start >= end ||
    end - start >= turns.length ||
    start < 0 ||
    end < 0 ||
    end === turns.length
  ) {
    throw new Error(
      "The 'turnWindow' passed is invalid. Please recheck your 'turnWindow' values.",
    );
  }
  return true;
}

const TURN_FIELDS: Partial<Record<MultiTurnParams, (turn: Turn) => unknown>> = {
  [MultiTurnParams.ROLE]: (turn) => turn.role,
  [MultiTurnParams.CONTENT]: (turn) => turn.content,
  [MultiTurnParams.METADATA]: (turn) => turn.additionalMetadata,
  [MultiTurnParams.RETRIEVAL_CONTEXT]: (turn) => turn.retrievalContext,
  [MultiTurnParams.TOOLS_CALLED]: (turn) => turn.toolsCalled,
  [MultiTurnParams.MCP_TOOLS]: (turn) => turn.mcpToolsCalled,
  [MultiTurnParams.MCP_RESOURCES]: (turn) => turn.mcpResourcesCalled,
  [MultiTurnParams.MCP_PROMPTS]: (turn) => turn.mcpPromptsCalled,
};

function formatTurnValue(value: unknown): string {
  if (value == null) return "None";
  if (typeof value === "string") return value;
  return JSON.stringify(value);
}

abstract class ConversationalBaseNode {
  readonly multiTurn = true;
  label?: string;
  evaluationParams?: MultiTurnParams[];
  turnWindow?: TurnWindow;

  validate(): void {}

  protected resolveText(ctx: NodeContext, sep = "\n\n"): string {
    const testCase = ctx.testCase as ConversationalTestCase;
    let text = "";
    for (const parent of ctx.parents ?? []) {
      if (parent.nodeKind === "task") {
        const task = parent as ConversationalTaskNode;
        text += `${task.outputLabel}:\n${ctx.outputs.get(parent)}${sep}`;
      }
    }
    if (this.evaluationParams == null) return text;

    let start = 0;
    let end = testCase.turns.length - 1;
    if (this.turnWindow != null) {
      isValidTurnWindow(this.turnWindow, testCase.turns);
      [start, end] = this.turnWindow;
    }
    text += "Full Conversation: \n";
    for (let index = start; index <= end; index++) {
      const turn = testCase.turns[index];
      for (const param of this.evaluationParams) {
        const value = TURN_FIELDS[param]?.(turn);
        text += `${CONVERSATIONAL_G_EVAL_PARAMS[param] ?? param}:\n${formatTurnValue(value)}\n\n`;
      }
    }
    return text;
  }
}

export interface ConversationalTaskNodeOptions {
  instructions: string;
  outputLabel: string;
  evaluationParams?: MultiTurnParams[];
  turnWindow?: TurnWindow;
  label?: string;
}

export class ConversationalTaskNode
  extends ConversationalBaseNode
  implements ExecutableNode
{
  readonly nodeKind = "task" as const;
  readonly children: AnyNode[] = [];
  instructions: string;
  outputLabel: string;

  constructor(options: ConversationalTaskNodeOptions) {
    super();
    this.instructions = options.instructions;
    this.outputLabel = options.outputLabel;
    this.evaluationParams = options.evaluationParams;
    this.turnWindow = options.turnWindow;
    this.label = options.label;
  }

  addNode<
    T extends
      | ConversationalTaskNode
      | ConversationalBinaryJudgementNode
      | ConversationalNonBinaryJudgementNode,
  >(child: T): T {
    this.children.push(child);
    return child;
  }

  validate(): void {
    for (const child of this.children) {
      if (child.nodeKind === "verdict") {
        throw new Error(
          "A ConversationalTaskNode must not have a ConversationalVerdictNode as one of their 'children'.",
        );
      }
    }
  }

  async execute(ctx: NodeContext): Promise<unknown> {
    if (this.evaluationParams == null && ctx.parents == null) {
      throw new Error(
        "A ConversationalTaskNode must have either a 'evaluationParams' or parent node(s).",
      );
    }
    const prompt = resolveTemplate(
      "metrics",
      "ConversationalTaskNode",
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
      "______________________________________________",
      "**********************************************",
      "ConversationalTaskNode",
      this.label,
      this.instructions,
      this.outputLabel,
      depth,
      result,
    );
  }
}

export interface ConversationalJudgementNodeOptions {
  criteria: string;
  evaluationParams?: MultiTurnParams[];
  turnWindow?: TurnWindow;
  label?: string;
}

export interface ConversationalAddVerdictOptions {
  score?: number;
  then?: AnyNode | ChildMetric;
}

export class ConversationalBinaryJudgementNode
  extends ConversationalBaseNode
  implements ExecutableNode
{
  readonly nodeKind = "binaryJudgement" as const;
  readonly children: ConversationalVerdictNode[] = [];
  criteria: string;

  constructor(options: ConversationalJudgementNodeOptions) {
    super();
    this.criteria = options.criteria;
    this.evaluationParams = options.evaluationParams;
    this.turnWindow = options.turnWindow;
    this.label = options.label;
  }

  addVerdict(
    verdict: boolean,
    options: ConversationalAddVerdictOptions = {},
  ): ConversationalVerdictNode {
    const node = new ConversationalVerdictNode({
      verdict,
      score: options.score,
      child: options.then,
    });
    this.children.push(node);
    return node;
  }

  validate(): void {
    if (this.children.length !== 2) {
      throw new Error(
        "ConversationalBinaryJudgementNode must have exactly 2 children.",
      );
    }
    const verdicts = this.children.map((child) => child.verdict);
    if (verdicts.some((verdict) => typeof verdict !== "boolean")) {
      throw new Error(
        "All children of ConversationalBinaryJudgementNode must have a boolean verdict.",
      );
    }
    if (
      verdicts.filter((verdict) => verdict === true).length !== 1 ||
      verdicts.filter((verdict) => verdict === false).length !== 1
    ) {
      throw new Error(
        "ConversationalBinaryJudgementNode must have one True and one False ConversationalVerdictNode child.",
      );
    }
  }

  async execute(ctx: NodeContext): Promise<JudgementVerdict> {
    const prompt = resolveTemplate(
      "metrics",
      "ConversationalBinaryJudgement",
      "generate_binary_verdict",
      { criteria: this.criteria, text: this.resolveText(ctx) },
    );
    return generateWithSchema(ctx.metric, prompt, BinaryJudgementVerdictSchema);
  }

  verboseLog(depth: number, result: unknown): string {
    return judgementVerboseLog(
      "ConversationalBinaryJudgementNode",
      34,
      48,
      this.label,
      this.criteria,
      depth,
      result as JudgementVerdict,
    );
  }
}

export class ConversationalNonBinaryJudgementNode
  extends ConversationalBaseNode
  implements ExecutableNode
{
  readonly nodeKind = "nonBinaryJudgement" as const;
  readonly children: ConversationalVerdictNode[] = [];
  criteria: string;
  private verdictOptions: string[] = [];

  constructor(options: ConversationalJudgementNodeOptions) {
    super();
    this.criteria = options.criteria;
    this.evaluationParams = options.evaluationParams;
    this.turnWindow = options.turnWindow;
    this.label = options.label;
  }

  addVerdict(
    verdict: string,
    options: ConversationalAddVerdictOptions = {},
  ): ConversationalVerdictNode {
    const node = new ConversationalVerdictNode({
      verdict,
      score: options.score,
      child: options.then,
    });
    this.children.push(node);
    return node;
  }

  validate(): void {
    if (this.children.length === 0) {
      throw new Error(
        "ConversationalNonBinaryJudgementNode must have at least one child.",
      );
    }
    const seen = new Set<string>();
    for (const child of this.children) {
      if (typeof child.verdict !== "string") {
        throw new Error(
          "The verdict attribute of all children must be a string.",
        );
      }
      if (seen.has(child.verdict)) {
        throw new Error(
          `Duplicate verdict found: ${child.verdict} in children of ConversationalNonBinaryJudgementNode.`,
        );
      }
      seen.add(child.verdict);
    }
    this.verdictOptions = [...seen];
  }

  async execute(ctx: NodeContext): Promise<JudgementVerdict> {
    const prompt = resolveTemplate(
      "metrics",
      "ConversationalNonBinaryJudgement",
      "generate_non_binary_verdict",
      {
        criteria: this.criteria,
        text: this.resolveText(ctx),
        options: this.verdictOptions,
        example_verdict_option: this.verdictOptions[0],
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
      "ConversationalNonBinaryJudgementNode",
      37,
      53,
      this.label,
      this.criteria,
      depth,
      result as JudgementVerdict,
    );
  }
}

export interface ConversationalVerdictNodeOptions {
  verdict: string | boolean;
  score?: number;
  child?: AnyNode | ChildMetric;
}

export class ConversationalVerdictNode
  extends ConversationalBaseNode
  implements VerdictLikeNode
{
  readonly nodeKind = "verdict" as const;
  readonly verdict: string | boolean;
  readonly score?: number;
  readonly child?: AnyNode | ChildMetric;

  constructor(options: ConversationalVerdictNodeOptions) {
    super();
    validateVerdictBranch(
      options.score,
      options.child,
      "ConversationalVerdictNode",
    );
    this.verdict = options.verdict;
    this.score = options.score;
    this.child = options.child;
  }

  async generateReason(metric: DagHostMetric): Promise<string> {
    const prompt = resolveTemplate(
      "metrics",
      "ConversationalVerdictNode",
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
    testCase: ConversationalTestCase,
  ): Promise<ChildMetric> {
    const copy = cloneChildMetric(this.child as ChildMetric, metric);
    await copy.measure(testCase);
    return copy;
  }

  verboseLog(depth: number, childMetric?: ChildMetric): string {
    return verdictVerboseLog(
      "_________________________________________________",
      "*************************************************",
      "ConversationalVerdictNode",
      this.verdict,
      depth,
      childMetric,
    );
  }
}
