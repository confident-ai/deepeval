import canonicalPrompt from "../prompts/evaluation.md";
import type { Ecosystem, EvaluationTarget } from "./deepeval.js";
import { judgeVariables, type JudgeProvider } from "./judge.js";

export const getCanonicalPrompt = (): string => canonicalPrompt.trim();

/** Name the judge without ever naming its value. */
const describeJudge = (provider: JudgeProvider | undefined): string =>
  provider
    ? `- Judge model: ${provider.label}, already configured through ${judgeVariables(
        provider,
      ).join(
        ", ",
      )} in the project dotenv. Let DeepEval read it; never open or echo those values.`
    : "- Judge model: none configured. Use judge-free metrics only and do not add LLM-judge metrics.";

const describeCloud = (useConfidentAi: boolean): string =>
  useConfidentAi
    ? "- Cloud: Confident AI. CONFIDENT_API_KEY is already configured for DeepEval. A completed result requires testRunId from the uploaded run."
    : "- Cloud: local only. Do not upload to Confident AI or use CONFIDENT_API_KEY. Omit testRunId.";

const SDK_LABELS: Record<Ecosystem, string> = {
  python: "the DeepEval Python SDK",
  node: "the DeepEval TypeScript SDK",
};

/**
 * The wizard resolved this by finding or installing the SDK, which is firmer
 * evidence than a manifest scan; without it the agent detects the language
 * itself, as the prompt already instructs.
 */
const describeSdk = (target: EvaluationTarget | undefined): string =>
  target === undefined
    ? ""
    : `- SDK: ${SDK_LABELS[target.sdk]}, installed for this project. ${
        target.shape === "component-level"
          ? "Instrument and evaluate in that language."
          : "This application is in neither language, so it cannot be instrumented: build the black-box evaluation described above and report the shape as black-box."
      }\n`;

export const buildAgentPrompt = (
  projectDirectory: string,
  resultFile: string,
  judge: { provider?: JudgeProvider } = {},
  useConfidentAi = true,
  target?: EvaluationTarget,
): string =>
  `${getCanonicalPrompt()}

Runtime context:
- Project directory: ${projectDirectory}
- Structured result destination: ${resultFile}
${describeSdk(target)}${describeJudge(judge.provider)}
${describeCloud(useConfidentAi)}

Work only in the project directory. The destination path is supplied for
clarity, but write it through CONFIDENT_SETUP_RESULT_FILE.`;
