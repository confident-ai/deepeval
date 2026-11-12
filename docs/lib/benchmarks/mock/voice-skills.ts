import type { Benchmark, BenchmarkJourney } from "../types";
import journeys from "./voice-skills-journeys.json";

export const voiceSkills: Benchmark = {
  slug: "voice-skills",
  name: "Voice Skills",
  category: "voice-agents",
  tagline: "Does the caller get what they rang for, or hang up?",
  summary:
    "Recorded callers with strong accents, bad lines and noisy rooms ring a voice agent about one real task. Each caller is a graph: a mistake by the agent sends the call down a harder path, and repeated mistakes end it.",
  version: "v0.1 preview",
  updatedAt: "2026-11-11",
  grading: {
    method: "mixed",
    summary:
      "Tool calls, the ending the call reached, timing and efficiency are checked in code. An LLM judge grades the skills that need judgement, one criterion each, and a model that hears audio grades voice style.",
    runsPerCase: 3,
  },
  dataset: {
    size: journeys.length,
    unit: "journeys",
    source: "Scripted caller graphs with a recording for every caller line",
    license: "Apache 2.0",
    host: "GitHub",
    url: "https://github.com/confident-ai/deepeval",
    publicShare: 1,
  },
  metrics: [
    {
      id: "pass3",
      label: "pass³",
      pickLabel: "Most reliable",
      unit: "percent",
      higherIsBetter: true,
      description:
        "Share of journeys that reach an expected ending and pass every metric on all three calls.",
    },
    {
      id: "efficiency",
      label: "Efficiency",
      pickLabel: "Most direct",
      unit: "percent",
      higherIsBetter: true,
      description:
        "The fewest caller turns the journey allows, divided by the turns the call took.",
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
  primaryMetricId: "pass3",
  tradeoffMetricId: "latency",
  segmentLabel: "Industry",
  segments: [
    { id: "airline", label: "Airline", caseCount: 2 },
    { id: "retail-bank", label: "Retail bank", caseCount: 2 },
    { id: "energy-supplier", label: "Energy supplier", caseCount: 2 },
    { id: "pharmacy", label: "Pharmacy", caseCount: 2 },
    { id: "broadband", label: "Broadband", caseCount: 2 },
    { id: "motor-insurance", label: "Motor insurance", caseCount: 2 },
    { id: "hotel", label: "Hotel", caseCount: 2 },
    { id: "parcel-delivery", label: "Parcel delivery", caseCount: 2 },
    { id: "roadside-assistance", label: "Roadside assistance", caseCount: 2 },
    { id: "online-retailer", label: "Online retailer", caseCount: 2 },
  ],
  skills: [
    {
      id: "listening",
      label: "Listening",
      description:
        "Hearing names and numbers through accents, noise, coughs and bad lines, and ignoring speech not meant for the agent.",
    },
    {
      id: "turn-taking",
      label: "Turn taking",
      description:
        "Waiting through pauses, holds and silence, carrying on through \"mm-hmm\", and stopping when cut off.",
    },
    {
      id: "task-handling",
      label: "Task handling",
      description:
        "Finishing the task through corrections, changed minds, failing tools and upset callers, without claiming false success.",
    },
    {
      id: "speaking",
      label: "Speaking",
      description:
        "Reading numbers back exactly, saying prices and abbreviations naturally, and following a voice style.",
    },
    {
      id: "boundaries",
      label: "Boundaries",
      description:
        "Holding the line on security checks, privacy, advice, emergencies and instructions hidden in what the caller dictates.",
    },
  ],
  entries: [],
  methodology: [
    {
      title: "The caller is a graph",
      body: "Every caller line is a node, and each edge is something the agent might do. A good agent takes the shortest route. An agent that mishears, re-asks or pushes back is corrected a limited number of times, then the caller gives up.",
    },
    {
      title: "Same caller for every agent",
      body: "Each line is a recording, so every agent hears the same voice, accent and pace. Line quality and background noise are laid over the recordings.",
    },
    {
      title: "Endings are the answer key",
      body: "A journey has expected endings and failure endings. A call fails if it reaches a failure ending, makes a tool call it must not make, or misses one it should.",
    },
    {
      title: "Graded skill by skill",
      body: "A call is graded only on the skills it tests, each with its own check, so a pass rate per skill is taken over the journeys that test it.",
    },
    {
      title: "Three calls per journey",
      body: "pass³ counts a journey only when all three calls pass, which separates reliable agents from lucky ones.",
    },
  ],
  runSnippet: `from deepeval.benchmarks import VoiceSkills
from deepeval.voice import LiveKitConnector, VoiceConfig

benchmark = VoiceSkills(dataset_path="voice_skills")
benchmark.evaluate(
    voice_config=VoiceConfig(connector=LiveKitConnector(agent_name="your-agent"))
)

print(benchmark.overall_score)
print(benchmark.skill_scores)`,
  faqs: [
    {
      question: "Why is the caller a graph and not a script?",
      answer:
        "A script carries on whatever the agent does. A graph lets the caller react: correct a wrong digit, get short after being asked twice, and hang up after a third time. That is what a real caller does, and it makes a weak agent's call longer and harder.",
    },
    {
      question: "Why are there no results yet?",
      answer:
        "This is a preview of the journeys and the grading. Models are added once their runs are complete and checked.",
    },
    {
      question: "Can I see exactly what the caller says?",
      answer:
        "Yes. Every journey is on this page with its graph, the recording of each caller line, the tool calls expected and the ones that must not happen.",
    },
  ],
  journeys: journeys as BenchmarkJourney[],
};
