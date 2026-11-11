import type { Benchmark } from "../types";

export const bigBenchAudio: Benchmark = {
  slug: "big-bench-audio",
  name: "Big Bench Audio",
  category: "speech-to-speech",
  tagline: "Does reasoning survive when the question is spoken?",
  summary:
    "1,000 reasoning questions from Big Bench Hard, recorded as audio. The model hears the question and answers out loud, which exposes the gap between a model's text reasoning and its speech reasoning.",
  version: "v1.1",
  updatedAt: "2026-10-30",
  grading: {
    method: "llm-judge",
    summary:
      "The spoken answer is transcribed, then an LLM judge compares it with the official answer and labels it correct or incorrect.",
    judgeModel: "claude-sonnet-5-5 (temperature 0)",
    runsPerCase: 3,
  },
  dataset: {
    size: 1000,
    unit: "audio questions",
    source: "Four Big Bench Hard categories, 250 questions each, synthesized with 23 voices",
    license: "MIT",
    host: "Hugging Face",
    url: "https://huggingface.co/datasets/ArtificialAnalysis/big_bench_audio",
    publicShare: 1,
  },
  metrics: [
    {
      id: "accuracy",
      label: "Accuracy",
      pickLabel: "Top ranked",
      unit: "percent",
      higherIsBetter: true,
      description: "Share of questions answered correctly from audio.",
    },
    {
      id: "textAccuracy",
      label: "Text baseline",
      unit: "percent",
      higherIsBetter: true,
      description: "Accuracy of the same model on the text version.",
    },
    {
      id: "ttfa",
      label: "First audio",
      pickLabel: "Fastest reply",
      unit: "seconds",
      higherIsBetter: false,
      description: "Median time to first audio in the reply.",
      decimals: 2,
    },
  ],
  primaryMetricId: "accuracy",
  tradeoffMetricId: "ttfa",
  segmentLabel: "Category",
  segments: [
    { id: "formal-fallacies", label: "Formal fallacies", caseCount: 250 },
    { id: "navigate", label: "Navigate", caseCount: 250 },
    { id: "object-counting", label: "Object counting", caseCount: 250 },
    { id: "web-of-lies", label: "Web of lies", caseCount: 250 },
  ],
  skills: [
    {
      id: "deduction",
      label: "Logical deduction",
      description: "Judging whether a spoken argument is valid.",
    },
    {
      id: "spatial",
      label: "Spatial tracking",
      description: "Following a sequence of spoken movement instructions.",
    },
    {
      id: "counting",
      label: "Counting",
      description: "Keeping a running count across a spoken list.",
    },
    {
      id: "truth-tracking",
      label: "Truth tracking",
      description: "Resolving chains of who is lying about whom.",
    },
    {
      id: "answer-clarity",
      label: "Answer clarity",
      description: "Stating one final answer the judge can extract.",
    },
  ],
  entries: [
    {
      model: "GPT Realtime 2.1",
      provider: "OpenAI",
      scores: { accuracy: 88.6, textAccuracy: 96.1, ttfa: 0.91 },
      segmentScores: {
        accuracy: { "formal-fallacies": 83.5, navigate: 89.7, "object-counting": 88.8, "web-of-lies": 85.5 },
        textAccuracy: { "formal-fallacies": 92.4, navigate: 97.3, "object-counting": 99.4, "web-of-lies": 93.8 },
        ttfa: { "formal-fallacies": 0.94, navigate: 0.91, "object-counting": 0.89, "web-of-lies": 0.93 },
      },
      skillScores: {
        deduction: 81,
        spatial: 94,
        counting: 86,
        "truth-tracking": 80,
        "answer-clarity": 97,
      },
      confidenceInterval: 1.1,
      evaluatedAt: "2026-10-26",
    },
    {
      model: "Gemini 3.1 Flash Live",
      provider: "Google",
      scores: { accuracy: 86.2, textAccuracy: 94.8, ttfa: 0.74 },
      segmentScores: {
        accuracy: { "formal-fallacies": 81.4, navigate: 87.7, "object-counting": 89.5, "web-of-lies": 80.9 },
        textAccuracy: { "formal-fallacies": 92.5, navigate: 98.0, "object-counting": 93.8, "web-of-lies": 89.8 },
        ttfa: { "formal-fallacies": 0.74, navigate: 0.75, "object-counting": 0.73, "web-of-lies": 0.74 },
      },
      skillScores: {
        deduction: 78,
        spatial: 90,
        counting: 88,
        "truth-tracking": 81,
        "answer-clarity": 86,
      },
      confidenceInterval: 1.2,
      evaluatedAt: "2026-10-26",
    },
    {
      model: "Grok Voice 3.0",
      provider: "xAI",
      scores: { accuracy: 83.9, textAccuracy: 93.5, ttfa: 0.68 },
      segmentScores: {
        accuracy: { "formal-fallacies": 78.0, navigate: 85.6, "object-counting": 88.8, "web-of-lies": 82.1 },
        textAccuracy: { "formal-fallacies": 92.9, navigate: 98.2, "object-counting": 97.9, "web-of-lies": 93.1 },
        ttfa: { "formal-fallacies": 0.69, navigate: 0.65, "object-counting": 0.66, "web-of-lies": 0.66 },
      },
      skillScores: {
        deduction: 75,
        spatial: 78,
        counting: 76,
        "truth-tracking": 77,
        "answer-clarity": 87,
      },
      confidenceInterval: 1.3,
      evaluatedAt: "2026-10-22",
    },
    {
      model: "Nova Sonic 2",
      provider: "Amazon",
      scores: { accuracy: 78.4, textAccuracy: 90.2, ttfa: 0.97 },
      segmentScores: {
        accuracy: { "formal-fallacies": 75.9, navigate: 84.5, "object-counting": 83.6, "web-of-lies": 78.1 },
        textAccuracy: { "formal-fallacies": 86.3, navigate: 92.4, "object-counting": 90.2, "web-of-lies": 89.7 },
        ttfa: { "formal-fallacies": 0.98, navigate: 0.95, "object-counting": 0.93, "web-of-lies": 0.98 },
      },
      skillScores: {
        deduction: 64,
        spatial: 74,
        counting: 72,
        "truth-tracking": 64,
        "answer-clarity": 72,
      },
      confidenceInterval: 1.5,
      evaluatedAt: "2026-10-12",
    },
    {
      model: "GPT Realtime 2.1 Mini",
      provider: "OpenAI",
      scores: { accuracy: 76.0, textAccuracy: 89.7, ttfa: 0.55 },
      segmentScores: {
        accuracy: { "formal-fallacies": 69.5, navigate: 78.3, "object-counting": 80.3, "web-of-lies": 77.2 },
        textAccuracy: { "formal-fallacies": 85.0, navigate: 92.5, "object-counting": 92.4, "web-of-lies": 88.9 },
        ttfa: { "formal-fallacies": 0.60, navigate: 0.56, "object-counting": 0.55, "web-of-lies": 0.55 },
      },
      skillScores: {
        deduction: 61,
        spatial: 73,
        counting: 73,
        "truth-tracking": 67,
        "answer-clarity": 70,
      },
      confidenceInterval: 1.5,
      evaluatedAt: "2026-10-26",
    },
    {
      model: "Phonic v1",
      provider: "Phonic",
      scores: { accuracy: 71.3, textAccuracy: 87.4, ttfa: 0.79 },
      segmentScores: {
        accuracy: { "formal-fallacies": 64.3, navigate: 73.4, "object-counting": 72.2, "web-of-lies": 67.2 },
        textAccuracy: { "formal-fallacies": 80.9, navigate: 87.0, "object-counting": 86.6, "web-of-lies": 81.4 },
        ttfa: { "formal-fallacies": 0.86, navigate: 0.77, "object-counting": 0.81, "web-of-lies": 0.81 },
      },
      skillScores: {
        deduction: 58,
        spatial: 67,
        counting: 66,
        "truth-tracking": 57,
        "answer-clarity": 68,
      },
      confidenceInterval: 1.6,
      evaluatedAt: "2026-10-12",
    },
    {
      model: "Higgs Realtime",
      provider: "Boson AI",
      scores: { accuracy: 63.8, textAccuracy: 82.0, ttfa: 0.86 },
      segmentScores: {
        accuracy: { "formal-fallacies": 60.9, navigate: 64.9, "object-counting": 67.9, "web-of-lies": 64.7 },
        textAccuracy: { "formal-fallacies": 82.7, navigate: 85.1, "object-counting": 81.7, "web-of-lies": 80.8 },
        ttfa: { "formal-fallacies": 0.91, navigate: 0.85, "object-counting": 0.86, "web-of-lies": 0.85 },
      },
      skillScores: {
        deduction: 54,
        spatial: 60,
        counting: 57,
        "truth-tracking": 49,
        "answer-clarity": 64,
      },
      confidenceInterval: 1.7,
      openWeights: true,
      evaluatedAt: "2026-09-30",
    },
    {
      model: "Moshi",
      provider: "Kyutai",
      scores: { accuracy: 41.5, textAccuracy: 61.9, ttfa: 0.31 },
      segmentScores: {
        accuracy: { "formal-fallacies": 35.3, navigate: 47.6, "object-counting": 47.8, "web-of-lies": 38.8 },
        textAccuracy: { "formal-fallacies": 54.3, navigate: 67.9, "object-counting": 68.9, "web-of-lies": 58.4 },
        ttfa: { "formal-fallacies": 0.33, navigate: 0.28, "object-counting": 0.30, "web-of-lies": 0.33 },
      },
      skillScores: {
        deduction: 45,
        spatial: 57,
        counting: 54,
        "truth-tracking": 43,
        "answer-clarity": 52,
      },
      confidenceInterval: 1.8,
      openWeights: true,
      evaluatedAt: "2026-09-30",
    },
  ],
  methodology: [
    {
      title: "Audio in, audio out",
      body: "The model receives only the recording. Its spoken reply is transcribed with a fixed speech-to-text model before grading.",
    },
    {
      title: "One judge, pinned",
      body: "A single judge model at temperature 0 sees the question, the official answer and the transcribed reply, and returns correct or incorrect.",
    },
    {
      title: "Text baseline",
      body: "Each model also answers the text version of the same questions, so the table shows how much reasoning is lost over speech.",
    },
    {
      title: "Three runs per question",
      body: "Accuracy is the mean of three independent runs.",
    },
  ],
  runSnippet: `from deepeval.benchmarks import BigBenchAudio

benchmark = BigBenchAudio()
benchmark.evaluate(model=your_speech_model)

print(benchmark.overall_score)`,
  faqs: [
    {
      question: "Why use an LLM judge for questions with fixed answers?",
      answer:
        "Spoken answers come wrapped in filler and explanation. A judge extracts the final answer more reliably than string matching on a transcript.",
    },
    {
      question: "Does transcription error count against the model?",
      answer:
        "The same speech-to-text model transcribes every reply, so any error applies equally across the table.",
    },
    {
      question: "Where is the audio?",
      answer:
        "All 1,000 recordings are on Hugging Face under an MIT license.",
    },
  ],
};
