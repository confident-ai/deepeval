import { NextResponse } from "next/server";
import { getMetric, rankEntries } from "@/lib/benchmarks";
import { getBenchmarks } from "@/lib/benchmarks/repository";

export const dynamic = "force-dynamic";

const CACHE_CONTROL = "public, s-maxage=60, stale-while-revalidate=300";

// Lists every published benchmark with its current leader. The full
// leaderboard for one benchmark is at /api/benchmarks/{slug}.
export async function GET() {
  const benchmarks = await getBenchmarks();

  const body = benchmarks.map((benchmark) => {
    const primary = getMetric(benchmark.metrics, benchmark.primaryMetricId);
    const [leader] = rankEntries(benchmark);
    return {
      slug: benchmark.slug,
      name: benchmark.name,
      category: benchmark.category,
      tagline: benchmark.tagline,
      version: benchmark.version,
      updatedAt: benchmark.updatedAt,
      primaryMetric: primary.id,
      models: benchmark.entries.length,
      leader: leader
        ? {
            model: leader.model,
            provider: leader.provider,
            score: leader.scores[primary.id],
          }
        : null,
      url: `/api/benchmarks/${benchmark.slug}`,
    };
  });

  return NextResponse.json(
    { benchmarks: body },
    { headers: { "Cache-Control": CACHE_CONTROL } },
  );
}
