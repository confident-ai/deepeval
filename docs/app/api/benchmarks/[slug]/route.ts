import { NextResponse, type NextRequest } from "next/server";
import { rankEntries } from "@/lib/benchmarks";
import { getBenchmark } from "@/lib/benchmarks/repository";

export const dynamic = "force-dynamic";

const CACHE_CONTROL = "public, s-maxage=60, stale-while-revalidate=300";

// One benchmark with its ranked leaderboard. The identifier is the
// benchmark's slug, or its exact name; see `getBenchmark` for how a match
// stays unique.
export async function GET(
  _request: NextRequest,
  context: RouteContext<"/api/benchmarks/[slug]">,
) {
  const { slug } = await context.params;
  const benchmark = await getBenchmark(slug);

  if (!benchmark) {
    return NextResponse.json(
      { error: `No published benchmark matches "${slug}".` },
      { status: 404 },
    );
  }

  return NextResponse.json(
    { ...benchmark, entries: rankEntries(benchmark) },
    { headers: { "Cache-Control": CACHE_CONTROL } },
  );
}
