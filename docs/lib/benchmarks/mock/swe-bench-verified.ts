import type { Benchmark } from "../types";

export const sweBenchVerified: Benchmark = {
  slug: "swe-bench-verified",
  name: "SWE-bench Verified",
  category: "ai-agents",
  tagline: "Can an AI agent resolve real GitHub issues?",
  summary:
    "500 human-validated software engineering tasks drawn from real GitHub issues. An AI agent gets a repository and an issue, and passes only if its patch makes the project's hidden tests pass.",
  version: "v1.2",
  updatedAt: "2026-11-03",
  grading: {
    method: "programmatic",
    summary:
      "Each patch is applied in a container and the repository's own unit tests are run. No model decides whether an answer is correct.",
    runsPerCase: 3,
  },
  dataset: {
    size: 500,
    unit: "tasks",
    source: "Real issues and pull requests from 12 open-source Python repositories",
    license: "MIT",
    host: "Hugging Face",
    url: "https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified",
    publicShare: 1,
  },
  metrics: [
    {
      id: "resolved",
      label: "Resolved",
      pickLabel: "Top ranked",
      unit: "percent",
      higherIsBetter: true,
      description: "Share of tasks where every target test passes.",
    },
    {
      id: "cost",
      label: "Cost / task",
      pickLabel: "Lowest cost",
      unit: "usd",
      higherIsBetter: false,
      description: "Mean API spend per task.",
      decimals: 2,
    },
    {
      id: "steps",
      label: "Steps",
      pickLabel: "Fewest steps",
      unit: "count",
      higherIsBetter: false,
      description: "Mean tool calls per task.",
      decimals: 0,
    },
  ],
  primaryMetricId: "resolved",
  tradeoffMetricId: "cost",
  segmentLabel: "Repository",
  segments: [
    { id: "django", label: "Django", caseCount: 231 },
    { id: "sympy", label: "SymPy", caseCount: 75 },
    { id: "scikit-learn", label: "scikit-learn", caseCount: 32 },
    { id: "other", label: "Other repositories", caseCount: 162 },
  ],
  skills: [
    {
      id: "localization",
      label: "Bug localization",
      description: "Finding the files and functions the issue is really about.",
    },
    {
      id: "correctness",
      label: "Patch correctness",
      description: "Writing a fix that passes the hidden tests without breaking others.",
    },
    {
      id: "test-awareness",
      label: "Test awareness",
      description: "Reproducing the bug and checking the fix before submitting.",
    },
    {
      id: "multi-file",
      label: "Multi-file edits",
      description: "Keeping a change consistent across several files.",
    },
    {
      id: "tool-use",
      label: "Tool use",
      description: "Using the shell, editor and search without wasted or failed calls.",
    },
  ],
  entries: [
    {
      model: "Claude Fable 5.2",
      provider: "Anthropic",
      scores: { resolved: 87.4, cost: 1.92, steps: 41 },
      segmentScores: {
        resolved: { django: 89.2, sympy: 78.5, "scikit-learn": 90.0, other: 82.9 },
        cost: { django: 1.85, sympy: 1.99, "scikit-learn": 1.95, other: 2.01 },
        steps: { django: 40, sympy: 45, "scikit-learn": 40, other: 43 },
      },
      skillScores: {
        localization: 90,
        correctness: 83,
        "test-awareness": 83,
        "multi-file": 81,
        "tool-use": 87,
      },
      confidenceInterval: 1.4,
      evaluatedAt: "2026-11-01",
    },
    {
      model: "GPT-6 Sol",
      provider: "OpenAI",
      scores: { resolved: 85.8, cost: 1.64, steps: 38 },
      segmentScores: {
        resolved: { django: 88.3, sympy: 76.3, "scikit-learn": 84.4, other: 83.8 },
        cost: { django: 1.57, sympy: 1.75, "scikit-learn": 1.70, other: 1.72 },
        steps: { django: 38, sympy: 40, "scikit-learn": 39, other: 38 },
      },
      skillScores: {
        localization: 88,
        correctness: 76,
        "test-awareness": 80,
        "multi-file": 77,
        "tool-use": 91,
      },
      confidenceInterval: 1.5,
      evaluatedAt: "2026-10-28",
    },
    {
      model: "Claude Opus 5.5",
      provider: "Anthropic",
      scores: { resolved: 84.6, cost: 2.31, steps: 44 },
      segmentScores: {
        resolved: { django: 88.1, sympy: 77.4, "scikit-learn": 89.4, other: 79.9 },
        cost: { django: 2.19, sympy: 2.44, "scikit-learn": 2.30, other: 2.46 },
        steps: { django: 44, sympy: 48, "scikit-learn": 42, other: 44 },
      },
      skillScores: {
        localization: 84,
        correctness: 73,
        "test-awareness": 74,
        "multi-file": 67,
        "tool-use": 80,
      },
      confidenceInterval: 1.6,
      evaluatedAt: "2026-10-21",
    },
    {
      model: "Gemini 3.1 Pro",
      provider: "Google",
      scores: { resolved: 82.2, cost: 1.18, steps: 47 },
      segmentScores: {
        resolved: { django: 87.4, sympy: 73.5, "scikit-learn": 84.3, other: 81.7 },
        cost: { django: 1.13, sympy: 1.24, "scikit-learn": 1.14, other: 1.18 },
        steps: { django: 46, sympy: 50, "scikit-learn": 46, other: 48 },
      },
      skillScores: {
        localization: 74,
        correctness: 71,
        "test-awareness": 69,
        "multi-file": 62,
        "tool-use": 74,
      },
      confidenceInterval: 1.7,
      evaluatedAt: "2026-10-19",
    },
    {
      model: "Claude Sonnet 5.5",
      provider: "Anthropic",
      scores: { resolved: 80.4, cost: 0.74, steps: 43 },
      segmentScores: {
        resolved: { django: 84.7, sympy: 73.4, "scikit-learn": 80.6, other: 79.5 },
        cost: { django: 0.74, sympy: 0.78, "scikit-learn": 0.71, other: 0.72 },
        steps: { django: 43, sympy: 44, "scikit-learn": 44, other: 44 },
      },
      skillScores: {
        localization: 70,
        correctness: 63,
        "test-awareness": 71,
        "multi-file": 63,
        "tool-use": 70,
      },
      confidenceInterval: 1.7,
      evaluatedAt: "2026-10-21",
    },
    {
      model: "Grok 4",
      provider: "xAI",
      scores: { resolved: 74.8, cost: 1.05, steps: 52 },
      segmentScores: {
        resolved: { django: 78.3, sympy: 68.5, "scikit-learn": 78.9, other: 74.9 },
        cost: { django: 1.01, sympy: 1.13, "scikit-learn": 0.98, other: 1.02 },
        steps: { django: 50, sympy: 53, "scikit-learn": 52, other: 51 },
      },
      skillScores: {
        localization: 64,
        correctness: 66,
        "test-awareness": 60,
        "multi-file": 56,
        "tool-use": 70,
      },
      confidenceInterval: 1.9,
      evaluatedAt: "2026-09-30",
    },
    {
      model: "Claude Haiku 5.5",
      provider: "Anthropic",
      scores: { resolved: 71.6, cost: 0.29, steps: 49 },
      segmentScores: {
        resolved: { django: 72.2, sympy: 65.0, "scikit-learn": 69.9, other: 71.3 },
        cost: { django: 0.28, sympy: 0.30, "scikit-learn": 0.30, other: 0.29 },
        steps: { django: 49, sympy: 53, "scikit-learn": 51, other: 51 },
      },
      skillScores: {
        localization: 64,
        correctness: 57,
        "test-awareness": 63,
        "multi-file": 50,
        "tool-use": 65,
      },
      confidenceInterval: 2.0,
      evaluatedAt: "2026-10-21",
    },
    {
      model: "DeepSeek V3",
      provider: "DeepSeek",
      scores: { resolved: 68.2, cost: 0.21, steps: 58 },
      segmentScores: {
        resolved: { django: 71.9, sympy: 62.3, "scikit-learn": 69.4, other: 69.1 },
        cost: { django: 0.21, sympy: 0.22, "scikit-learn": 0.21, other: 0.20 },
        steps: { django: 54, sympy: 61, "scikit-learn": 58, other: 56 },
      },
      skillScores: {
        localization: 61,
        correctness: 51,
        "test-awareness": 56,
        "multi-file": 43,
        "tool-use": 60,
      },
      confidenceInterval: 2.1,
      openWeights: true,
      evaluatedAt: "2026-09-12",
    },
    {
      model: "Qwen3 Coder",
      provider: "Alibaba",
      scores: { resolved: 66.0, cost: 0.18, steps: 61 },
      segmentScores: {
        resolved: { django: 70.0, sympy: 63.0, "scikit-learn": 69.8, other: 63.0 },
        cost: { django: 0.17, sympy: 0.18, "scikit-learn": 0.18, other: 0.19 },
        steps: { django: 59, sympy: 64, "scikit-learn": 58, other: 64 },
      },
      skillScores: {
        localization: 50,
        correctness: 48,
        "test-awareness": 45,
        "multi-file": 42,
        "tool-use": 50,
      },
      confidenceInterval: 2.1,
      openWeights: true,
      evaluatedAt: "2026-09-12",
    },
  ],
  methodology: [
    {
      title: "Same harness for every model",
      body: "Every model runs in the same agent loop with the same tools: a shell, a file editor and a search tool. Only the model changes between runs.",
    },
    {
      title: "Isolated containers",
      body: "Each task starts from a clean container pinned to the repository commit the issue was filed against.",
    },
    {
      title: "Hidden tests decide",
      body: "The agent never sees the tests that grade it. A task counts as resolved when the failing tests pass and no passing test breaks.",
    },
    {
      title: "Three runs per task",
      body: "Scores are the mean of three runs, with a 95% interval from bootstrap resampling over tasks.",
    },
  ],
  runSnippet: `from deepeval.benchmarks import SWEBenchVerified

benchmark = SWEBenchVerified(n_tasks=500)
benchmark.evaluate(model=your_agent)

print(benchmark.overall_score)`,
  faqs: [
    {
      question: "Why the Verified subset?",
      answer:
        "Human annotators removed tasks with unclear issue text or tests that reject valid fixes, so a failure reflects the agent and not the task.",
    },
    {
      question: "Is an LLM judge involved?",
      answer:
        "No. Grading is the repository's own test suite, run in a container.",
    },
    {
      question: "Can I reproduce a row?",
      answer:
        "Yes. The dataset is public and the snippet above runs the same harness against any model you can call from DeepEval.",
    },
  ],
};
