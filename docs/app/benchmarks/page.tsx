import type { Metadata } from "next";
import {
  formatCount,
  formatMetric,
  getMetric,
  rankEntries,
  type MethodologyStep,
} from "@/lib/benchmarks";
import { getBenchmarks } from "@/lib/benchmarks/repository";
import BenchmarksHero from "@/src/sections/benchmarks/BenchmarksHero";
import BenchmarkBrowser from "@/src/sections/benchmarks/BenchmarkBrowser";
import {
  BenchmarkSection,
  StepGrid,
} from "@/src/sections/benchmarks/BenchmarkSection";

// Results are read from the database on each request.
export const dynamic = "force-dynamic";

export const metadata: Metadata = {
  title: "Benchmarks",
  description:
    "Open benchmarks for AI agents, voice models and language models. Public datasets, published grading and leaderboards you can reproduce with DeepEval.",
  alternates: { canonical: "/benchmarks" },
};

const PRINCIPLES: MethodologyStep[] = [
  {
    title: "Open datasets",
    body: "Test sets are published on Hugging Face or GitHub with their licenses. Where a slice is held out, the page says how much and why.",
  },
  {
    title: "Grading you can read",
    body: "Each benchmark states whether scoring is programmatic, an LLM judge or both, and which judge model is pinned.",
  },
  {
    title: "Repeated runs",
    body: "Cases are run more than once and scores carry a 95% interval, so a one-point gap is not mistaken for a win.",
  },
  {
    title: "One command to reproduce",
    body: "Every leaderboard row comes from a DeepEval benchmark class. Run the same class against your own model to get a comparable number.",
  },
];

export default async function BenchmarksPage() {
  const benchmarks = await getBenchmarks();

  // A model is its provider plus its name.
  const models = new Set(
    benchmarks.flatMap((benchmark) =>
      benchmark.entries.map((entry) => `${entry.provider}/${entry.model}`),
    ),
  );
  const providers = new Set(
    benchmarks.flatMap((benchmark) =>
      benchmark.entries.map((entry) => entry.provider),
    ),
  );
  const cases = benchmarks.reduce(
    (total, benchmark) => total + benchmark.dataset.size,
    0,
  );

  // Benchmarks without published results have no winner to show.
  const winners = benchmarks.flatMap((benchmark) => {
    const primary = getMetric(benchmark.metrics, benchmark.primaryMetricId);
    const [leader] = rankEntries(benchmark);
    if (!leader) return [];
    return [
      {
        slug: benchmark.slug,
        benchmark: benchmark.name,
        metric: primary.label,
        score: formatMetric(leader.scores[primary.id], primary),
        model: leader.model,
        provider: leader.provider,
      },
    ];
  });

  return (
    <>
      <BenchmarksHero
        winners={winners}
        stats={[
          { label: "Benchmarks", value: String(benchmarks.length) },
          { label: "Models evaluated", value: String(models.size) },
          { label: "Providers", value: String(providers.size) },
          { label: "Test cases", value: formatCount(cases) },
        ]}
      />

      <BenchmarkSection
        id="benchmarks"
        label="Leaderboards"
        title="Pick a benchmark"
        description="Filter by what is being measured. Each benchmark has its own leaderboard, methodology and dataset page."
        fullWidth
      >
        <BenchmarkBrowser benchmarks={benchmarks} />
      </BenchmarkSection>

      <BenchmarkSection
        label="Principles"
        title="How the benchmarks are run"
        description="The same four rules apply to every leaderboard, whatever is being measured."
      >
        <StepGrid steps={PRINCIPLES} />
      </BenchmarkSection>
    </>
  );
}
