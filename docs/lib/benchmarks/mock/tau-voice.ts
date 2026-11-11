import type { Benchmark } from "../types";

export const tauVoice: Benchmark = {
  slug: "tau-voice",
  name: "τ-Voice",
  category: "voice-agents",
  tagline: "Can a voice agent finish the task on a live call?",
  summary:
    "Full-duplex customer service calls in retail, airline and telecom domains. A simulated caller speaks to the voice agent, and the call passes only if the agent follows policy and leaves the backend in the right state.",
  version: "v1.0",
  updatedAt: "2026-10-27",
  grading: {
    method: "mixed",
    summary:
      "Task success is checked in code against the final database state. An LLM judge reviews the transcript for policy violations the state check cannot see.",
    judgeModel: "gpt-4.1 (temperature 0)",
    runsPerCase: 3,
  },
  dataset: {
    size: 278,
    unit: "scenarios",
    source: "Scripted tasks with simulated callers across three domains",
    license: "MIT",
    host: "GitHub",
    url: "https://github.com/sierra-research/tau2-bench",
    publicShare: 1,
  },
  metrics: [
    {
      id: "pass1",
      label: "Task success",
      pickLabel: "Top ranked",
      unit: "percent",
      higherIsBetter: true,
      description: "Share of calls that end in the correct backend state.",
    },
    {
      id: "pass3",
      label: "pass³",
      pickLabel: "Most reliable",
      unit: "percent",
      higherIsBetter: true,
      description: "Share of scenarios that pass on all three calls.",
    },
    {
      id: "latency",
      label: "Reply latency",
      pickLabel: "Fastest reply",
      unit: "seconds",
      higherIsBetter: false,
      description: "Median time from end of caller speech to first agent audio.",
      decimals: 2,
    },
  ],
  primaryMetricId: "pass1",
  tradeoffMetricId: "latency",
  segmentLabel: "Domain",
  segments: [
    { id: "retail", label: "Retail", caseCount: 114 },
    { id: "airline", label: "Airline", caseCount: 50 },
    { id: "telecom", label: "Telecom", caseCount: 114 },
  ],
  skills: [
    {
      id: "listening",
      label: "Listening",
      description: "Catching names, numbers and addresses the caller actually said.",
    },
    {
      id: "turn-taking",
      label: "Turn taking",
      description: "Handling pauses and interruptions without talking over the caller.",
    },
    {
      id: "tool-calling",
      label: "Tool calling",
      description: "Calling the right backend tool with the right arguments mid-call.",
    },
    {
      id: "policy",
      label: "Policy adherence",
      description: "Refusing or escalating when the domain policy requires it.",
    },
    {
      id: "speaking",
      label: "Speaking",
      description: "Clear, natural speech with correct pronunciation of entities.",
    },
  ],
  entries: [
    {
      model: "GPT Realtime 2.1",
      provider: "OpenAI",
      scores: { pass1: 68.4, pass3: 49.2, latency: 0.82 },
      segmentScores: {
        pass1: { retail: 69.7, airline: 58.3, telecom: 69.3 },
        pass3: { retail: 51.4, airline: 43.2, telecom: 51.3 },
        latency: { retail: 0.83, airline: 0.88, telecom: 0.81 },
      },
      skillScores: {
        listening: 80,
        "turn-taking": 87,
        "tool-calling": 84,
        policy: 85,
        speaking: 90,
      },
      confidenceInterval: 2.6,
      evaluatedAt: "2026-10-24",
    },
    {
      model: "Gemini 3.1 Flash Live",
      provider: "Google",
      scores: { pass1: 64.9, pass3: 44.6, latency: 0.71 },
      segmentScores: {
        pass1: { retail: 68.5, airline: 58.2, telecom: 66.6 },
        pass3: { retail: 46.5, airline: 43.1, telecom: 44.9 },
        latency: { retail: 0.69, airline: 0.78, telecom: 0.72 },
      },
      skillScores: {
        listening: 81,
        "turn-taking": 88,
        "tool-calling": 78,
        policy: 75,
        speaking: 88,
      },
      confidenceInterval: 2.7,
      evaluatedAt: "2026-10-24",
    },
    {
      model: "Grok Voice 3.0",
      provider: "xAI",
      scores: { pass1: 61.3, pass3: 40.1, latency: 0.64 },
      segmentScores: {
        pass1: { retail: 68.0, airline: 57.5, telecom: 57.9 },
        pass3: { retail: 44.2, airline: 38.5, telecom: 39.2 },
        latency: { retail: 0.59, airline: 0.68, telecom: 0.68 },
      },
      skillScores: {
        listening: 69,
        "turn-taking": 76,
        "tool-calling": 72,
        policy: 70,
        speaking: 85,
      },
      confidenceInterval: 2.8,
      evaluatedAt: "2026-10-20",
    },
    {
      model: "GPT Realtime 2.1 Mini",
      provider: "OpenAI",
      scores: { pass1: 55.7, pass3: 33.8, latency: 0.58 },
      segmentScores: {
        pass1: { retail: 58.0, airline: 45.2, telecom: 54.1 },
        pass3: { retail: 35.0, airline: 30.3, telecom: 32.5 },
        latency: { retail: 0.58, airline: 0.68, telecom: 0.61 },
      },
      skillScores: {
        listening: 66,
        "turn-taking": 74,
        "tool-calling": 74,
        policy: 67,
        speaking: 78,
      },
      confidenceInterval: 2.9,
      evaluatedAt: "2026-10-24",
    },
    {
      model: "Phonic v1",
      provider: "Phonic",
      scores: { pass1: 52.2, pass3: 30.5, latency: 0.77 },
      segmentScores: {
        pass1: { retail: 57.0, airline: 46.4, telecom: 48.1 },
        pass3: { retail: 32.2, airline: 29.1, telecom: 28.6 },
        latency: { retail: 0.75, airline: 0.85, telecom: 0.80 },
      },
      skillScores: {
        listening: 66,
        "turn-taking": 71,
        "tool-calling": 68,
        policy: 63,
        speaking: 72,
      },
      confidenceInterval: 2.9,
      evaluatedAt: "2026-10-11",
    },
    {
      model: "Nova Sonic 2",
      provider: "Amazon",
      scores: { pass1: 49.6, pass3: 27.4, latency: 0.93 },
      segmentScores: {
        pass1: { retail: 52.9, airline: 39.8, telecom: 49.5 },
        pass3: { retail: 27.5, airline: 23.9, telecom: 27.5 },
        latency: { retail: 0.91, airline: 1.06, telecom: 0.90 },
      },
      skillScores: {
        listening: 52,
        "turn-taking": 58,
        "tool-calling": 55,
        policy: 51,
        speaking: 66,
      },
      confidenceInterval: 3.0,
      evaluatedAt: "2026-10-11",
    },
    {
      model: "Grok Voice Think Fast 2.0",
      provider: "xAI",
      scores: { pass1: 47.1, pass3: 24.9, latency: 0.49 },
      segmentScores: {
        pass1: { retail: 48.0, airline: 36.6, telecom: 43.7 },
        pass3: { retail: 25.8, airline: 20.7, telecom: 24.4 },
        latency: { retail: 0.47, airline: 0.56, telecom: 0.52 },
      },
      skillScores: {
        listening: 47,
        "turn-taking": 56,
        "tool-calling": 48,
        policy: 53,
        speaking: 63,
      },
      confidenceInterval: 3.0,
      evaluatedAt: "2026-10-20",
    },
    {
      model: "Higgs Realtime",
      provider: "Boson AI",
      scores: { pass1: 38.5, pass3: 17.6, latency: 0.88 },
      segmentScores: {
        pass1: { retail: 40.0, airline: 29.8, telecom: 36.4 },
        pass3: { retail: 17.5, airline: 14.7, telecom: 17.0 },
        latency: { retail: 0.88, airline: 1.05, telecom: 0.92 },
      },
      skillScores: {
        listening: 44,
        "turn-taking": 48,
        "tool-calling": 51,
        policy: 48,
        speaking: 56,
      },
      confidenceInterval: 3.1,
      openWeights: true,
      evaluatedAt: "2026-09-29",
    },
  ],
  methodology: [
    {
      title: "Simulated callers",
      body: "Each scenario gives a caller a goal, a persona and facts to reveal. The caller is rendered to speech and holds a real two-way call with the agent.",
    },
    {
      title: "State is the answer key",
      body: "Every scenario has an expected final database state. A call passes when the agent's tool calls produce exactly that state.",
    },
    {
      title: "Policy review",
      body: "A pinned LLM judge reads the transcript against the domain policy and fails calls that reach the right state the wrong way.",
    },
    {
      title: "Three calls per scenario",
      body: "pass³ counts a scenario only when all three calls pass, which separates reliable agents from lucky ones.",
    },
  ],
  runSnippet: `from deepeval.benchmarks import TauVoice
from deepeval.benchmarks.tasks import TauVoiceTask

benchmark = TauVoice(tasks=[TauVoiceTask.RETAIL, TauVoiceTask.AIRLINE])
benchmark.evaluate(model=your_voice_agent)

print(benchmark.overall_score)`,
  faqs: [
    {
      question: "Why are scores lower than text agent benchmarks?",
      answer:
        "The same tasks get harder over audio. Names, order numbers and addresses have to survive speech recognition, and the agent has to handle interruptions while it calls tools.",
    },
    {
      question: "What does the LLM judge decide?",
      answer:
        "Only policy compliance. Whether the task was completed is decided in code from the database state.",
    },
    {
      question: "Are cascaded pipelines included?",
      answer:
        "This leaderboard lists speech-to-speech models. Cascaded pipelines run on the same scenarios and are reported separately.",
    },
  ],
};
