import type {
  Benchmark,
  BenchmarkJourney,
  LeaderboardEntry,
  SkillFamily,
} from "../types";
import families from "./voice-skills-families.json";
import journeys from "./voice-skills-journeys.json";
import results from "./voice-skills-results.json";

export const voiceSkills: Benchmark = {
  slug: "voice-skills",
  name: "DeepEval VoiceBench Hard",
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
    runsPerCase: 1,
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
      id: "overall",
      label: "Overall score",
      pickLabel: "Best overall",
      unit: "percent",
      higherIsBetter: true,
      description:
        "The average of the five skill families. Each family is the average of its skills, and each skill is its average score over the journeys that test it.",
      decimals: 1,
    },
    {
      id: "completion",
      label: "Calls completed",
      pickLabel: "Most calls completed",
      unit: "percent",
      higherIsBetter: true,
      description:
        "Share of calls that reached an expected ending without the caller giving up or the agent crossing a line.",
      decimals: 1,
    },
    {
      id: "tools",
      label: "Tool accuracy",
      pickLabel: "Most accurate tool calls",
      unit: "percent",
      higherIsBetter: true,
      description:
        "Share of expected tool calls made with the right values, with nothing forbidden called.",
      decimals: 1,
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
    {
      id: "cost",
      label: "Cost per call",
      pickLabel: "Cheapest call",
      unit: "usd",
      higherIsBetter: false,
      description:
        "What the model cost for one call, averaged over the journeys. Estimated from the length of the audio on each side at the provider's list prices.",
      decimals: 3,
    },
    {
      id: "costPerMinute",
      label: "Cost per minute",
      unit: "usd",
      higherIsBetter: false,
      description: "Model cost divided by the length of the call.",
      decimals: 3,
    },
  ],
  primaryMetricId: "overall",
  tradeoffMetricId: "latency",
  segmentLabel: "Line quality",
  segments: [
    ["phone", "Phone line"],
    ["landline", "Landline"],
    ["speakerphone", "Speakerphone"],
    ["mobile", "Weak mobile"],
  ].map(([id, label]) => ({
    id,
    label,
    caseCount: journeys.filter((journey) => journey.segmentId === id).length,
  })),

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
  entries: results as unknown as LeaderboardEntry[],
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
      title: "One reference agent, many models",
      body: "Each model runs as the same agent: the same instructions, and the policy, records and tools of the journey it is on. Only the model changes.",
    },
    {
      title: "One call per journey, for now",
      body: "These first results are one call per journey. Three calls each, with a journey counted only when all three pass, come next.",
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
      question: "Why are the scores so low?",
      answer:
        "The callers are hard on purpose: strong accents, weak lines, noise, changed minds and requests the agent must refuse. Few calls are completed, so the overall score, which averages the skills, is what separates models today.",
    },
    {
      question: "Can I see exactly what the caller says?",
      answer:
        "Yes. Every journey is on this page with its graph, the recording of each caller line, the tool calls expected and the ones that must not happen.",
    },
  ],
  journeys: journeys as BenchmarkJourney[],
  skillFamilies: families as SkillFamily[],
};
