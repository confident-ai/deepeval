import type { Benchmark } from "../types";

export const humanitysLastExam: Benchmark = {
  slug: "humanitys-last-exam",
  name: "Humanity's Last Exam",
  category: "language-models",
  tagline: "How close are language models to expert-level knowledge?",
  summary:
    "2,500 expert-written questions across mathematics, the sciences and the humanities, built to sit at the edge of what specialists know. Answers are short and closed-ended, so grading stays unambiguous.",
  version: "v1.0",
  updatedAt: "2026-10-22",
  grading: {
    method: "llm-judge",
    summary:
      "An LLM judge extracts the model's final answer and checks it against the answer key. The judge does not see which model produced the answer.",
    judgeModel: "gpt-4.1 (temperature 0)",
    runsPerCase: 1,
  },
  dataset: {
    size: 2500,
    unit: "questions",
    source: "Written and reviewed by subject-matter experts",
    license: "MIT",
    host: "Hugging Face",
    url: "https://huggingface.co/datasets/cais/hle",
    publicShare: 0.9,
  },
  metrics: [
    {
      id: "accuracy",
      label: "Accuracy",
      pickLabel: "Top ranked",
      unit: "percent",
      higherIsBetter: true,
      description: "Share of questions answered correctly.",
    },
    {
      id: "calibration",
      label: "Calibration error",
      pickLabel: "Best calibrated",
      unit: "percent",
      higherIsBetter: false,
      description: "Gap between stated confidence and actual accuracy.",
    },
    {
      id: "cost",
      label: "Cost / question",
      pickLabel: "Lowest cost",
      unit: "usd",
      higherIsBetter: false,
      description: "Mean API spend per question.",
      decimals: 2,
    },
  ],
  primaryMetricId: "accuracy",
  tradeoffMetricId: "cost",
  segmentLabel: "Subject",
  segments: [
    { id: "math", label: "Mathematics", caseCount: 1025 },
    { id: "bio-med", label: "Biology and medicine", caseCount: 275 },
    { id: "physics", label: "Physics", caseCount: 225 },
    { id: "cs-ai", label: "Computer science and AI", caseCount: 250 },
    { id: "humanities", label: "Humanities and social science", caseCount: 725 },
  ],
  skills: [
    {
      id: "reasoning",
      label: "Multi-step reasoning",
      description: "Chaining several inferences to reach an answer.",
    },
    {
      id: "recall",
      label: "Specialist recall",
      description: "Retrieving facts only a domain expert would know.",
    },
    {
      id: "calculation",
      label: "Calculation",
      description: "Carrying exact arithmetic and algebra through to the end.",
    },
    {
      id: "figures",
      label: "Reading figures",
      description: "Questions that depend on a diagram, chart or image.",
    },
    {
      id: "calibration",
      label: "Calibration",
      description: "Stating confidence that matches how often it is right.",
    },
  ],
  entries: [
    {
      model: "Claude Fable 5.2",
      provider: "Anthropic",
      scores: { accuracy: 58.3, calibration: 31.2, cost: 0.42 },
      segmentScores: {
        accuracy: { math: 58.4, "bio-med": 64.7, physics: 54.8, "cs-ai": 60.7, humanities: 59.9 },
        calibration: { math: 30.2, "bio-med": 28.7, physics: 31.3, "cs-ai": 29.4, humanities: 31.4 },
        cost: { math: 0.41, "bio-med": 0.40, physics: 0.43, "cs-ai": 0.40, humanities: 0.42 },
      },
      skillScores: {
        reasoning: 80,
        recall: 87,
        calculation: 78,
        figures: 80,
        calibration: 89,
      },
      confidenceInterval: 1.9,
      evaluatedAt: "2026-10-20",
    },
    {
      model: "GPT-6 Sol",
      provider: "OpenAI",
      scores: { accuracy: 56.9, calibration: 34.8, cost: 0.38 },
      segmentScores: {
        accuracy: { math: 56.7, "bio-med": 58.5, physics: 57.4, "cs-ai": 63.0, humanities: 56.4 },
        calibration: { math: 33.8, "bio-med": 34.6, physics: 33.6, "cs-ai": 32.5, humanities: 34.2 },
        cost: { math: 0.37, "bio-med": 0.39, physics: 0.39, "cs-ai": 0.35, humanities: 0.39 },
      },
      skillScores: {
        reasoning: 83,
        recall: 89,
        calculation: 74,
        figures: 78,
        calibration: 80,
      },
      confidenceInterval: 1.9,
      evaluatedAt: "2026-10-20",
    },
    {
      model: "Gemini 3.1 Pro",
      provider: "Google",
      scores: { accuracy: 54.1, calibration: 38.5, cost: 0.27 },
      segmentScores: {
        accuracy: { math: 51.0, "bio-med": 61.5, physics: 55.4, "cs-ai": 56.7, humanities: 55.6 },
        calibration: { math: 39.0, "bio-med": 35.3, physics: 39.0, "cs-ai": 37.9, humanities: 36.7 },
        cost: { math: 0.29, "bio-med": 0.24, physics: 0.26, "cs-ai": 0.27, humanities: 0.26 },
      },
      skillScores: {
        reasoning: 74,
        recall: 77,
        calculation: 69,
        figures: 66,
        calibration: 78,
      },
      confidenceInterval: 2.0,
      evaluatedAt: "2026-10-15",
    },
    {
      model: "Claude Opus 5.5",
      provider: "Anthropic",
      scores: { accuracy: 51.6, calibration: 33.0, cost: 0.51 },
      segmentScores: {
        accuracy: { math: 45.2, "bio-med": 56.0, physics: 50.2, "cs-ai": 53.2, humanities: 52.4 },
        calibration: { math: 35.1, "bio-med": 30.3, physics: 32.4, "cs-ai": 32.6, humanities: 32.0 },
        cost: { math: 0.56, "bio-med": 0.48, physics: 0.52, "cs-ai": 0.52, humanities: 0.50 },
      },
      skillScores: {
        reasoning: 69,
        recall: 73,
        calculation: 62,
        figures: 67,
        calibration: 73,
      },
      confidenceInterval: 2.0,
      evaluatedAt: "2026-10-15",
    },
    {
      model: "Grok 4",
      provider: "xAI",
      scores: { accuracy: 44.7, calibration: 46.2, cost: 0.24 },
      segmentScores: {
        accuracy: { math: 45.0, "bio-med": 45.9, physics: 42.1, "cs-ai": 46.5, humanities: 48.7 },
        calibration: { math: 46.3, "bio-med": 46.8, physics: 46.7, "cs-ai": 43.9, humanities: 45.3 },
        cost: { math: 0.25, "bio-med": 0.23, physics: 0.24, "cs-ai": 0.24, humanities: 0.24 },
      },
      skillScores: {
        reasoning: 60,
        recall: 64,
        calculation: 60,
        figures: 61,
        calibration: 68,
      },
      confidenceInterval: 1.9,
      evaluatedAt: "2026-09-28",
    },
    {
      model: "Claude Sonnet 5.5",
      provider: "Anthropic",
      scores: { accuracy: 42.2, calibration: 37.4, cost: 0.16 },
      segmentScores: {
        accuracy: { math: 37.5, "bio-med": 43.7, physics: 44.1, "cs-ai": 47.7, humanities: 45.6 },
        calibration: { math: 39.2, "bio-med": 37.9, physics: 37.5, "cs-ai": 35.0, humanities: 35.5 },
        cost: { math: 0.17, "bio-med": 0.15, physics: 0.16, "cs-ai": 0.15, humanities: 0.16 },
      },
      skillScores: {
        reasoning: 53,
        recall: 58,
        calculation: 57,
        figures: 51,
        calibration: 55,
      },
      confidenceInterval: 1.9,
      evaluatedAt: "2026-10-15",
    },
    {
      model: "DeepSeek V3",
      provider: "DeepSeek",
      scores: { accuracy: 29.8, calibration: 58.1, cost: 0.04 },
      segmentScores: {
        accuracy: { math: 29.9, "bio-med": 34.7, physics: 30.9, "cs-ai": 31.9, humanities: 34.3 },
        calibration: { math: 56.9, "bio-med": 54.4, physics: 57.3, "cs-ai": 54.9, humanities: 52.0 },
        cost: { math: 0.04, "bio-med": 0.04, physics: 0.04, "cs-ai": 0.04, humanities: 0.04 },
      },
      skillScores: {
        reasoning: 48,
        recall: 61,
        calculation: 49,
        figures: 44,
        calibration: 54,
      },
      confidenceInterval: 1.8,
      openWeights: true,
      evaluatedAt: "2026-09-14",
    },
    {
      model: "Llama 4 Maverick",
      provider: "Meta",
      scores: { accuracy: 21.4, calibration: 66.3, cost: 0.03 },
      segmentScores: {
        accuracy: { math: 21.4, "bio-med": 23.8, physics: 17.8, "cs-ai": 26.6, humanities: 21.6 },
        calibration: { math: 66.9, "bio-med": 61.1, physics: 76.5, "cs-ai": 56.8, humanities: 63.4 },
        cost: { math: 0.03, "bio-med": 0.03, physics: 0.03, "cs-ai": 0.03, humanities: 0.03 },
      },
      skillScores: {
        reasoning: 43,
        recall: 48,
        calculation: 40,
        figures: 38,
        calibration: 47,
      },
      confidenceInterval: 1.6,
      openWeights: true,
      evaluatedAt: "2026-09-14",
    },
  ],
  methodology: [
    {
      title: "No tools, no search",
      body: "Models answer from their own knowledge with reasoning enabled at each provider's default setting.",
    },
    {
      title: "Blind judging",
      body: "The judge sees the question, the answer key and the response, with the model's name removed.",
    },
    {
      title: "Confidence is scored too",
      body: "Each model states a confidence with its answer. Calibration error measures how far that confidence sits from its real accuracy.",
    },
    {
      title: "Held-out slice",
      body: "A private portion of the questions is scored alongside the public set to check that results are not inflated by training-data overlap.",
    },
  ],
  runSnippet: `from deepeval.benchmarks import HumanitysLastExam

benchmark = HumanitysLastExam()
benchmark.evaluate(model=your_model)

print(benchmark.overall_score)`,
  faqs: [
    {
      question: "Why is part of the dataset private?",
      answer:
        "Public questions end up in training data over time. A held-out slice shows whether a model's public score still holds on questions it cannot have seen.",
    },
    {
      question: "How reliable is the LLM judge?",
      answer:
        "Answers are short and closed-ended, so the judge is matching against a key, not rating quality. The judge model and temperature are pinned.",
    },
    {
      question: "Why report calibration?",
      answer:
        "A model that is wrong and confident is riskier in production than one that is wrong and unsure. Accuracy alone hides that difference.",
    },
  ],
};
