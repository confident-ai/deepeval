import type { Prisma } from "@/generated/prisma/client";
import type {
  BenchmarkCategory as DbCategory,
  DatasetHost as DbHost,
  GradingMethod as DbGrading,
  MetricUnit as DbUnit,
} from "@/generated/prisma/enums";
import { getDb } from "@/lib/db";
import { MOCK_BENCHMARKS } from "./mock";
import type {
  Benchmark,
  BenchmarkCategory,
  BenchmarkDataset,
  BenchmarkMetric,
  GradingMethod,
  LeaderboardEntry,
  MetricUnit,
} from "./types";

const CATEGORIES: Record<DbCategory, BenchmarkCategory> = {
  AI_AGENTS: "ai-agents",
  VOICE_AGENTS: "voice-agents",
  SPEECH_TO_SPEECH: "speech-to-speech",
  SPEECH_TO_TEXT: "speech-to-text",
  LANGUAGE_MODELS: "language-models",
};

const GRADING_METHODS: Record<DbGrading, GradingMethod> = {
  PROGRAMMATIC: "programmatic",
  LLM_JUDGE: "llm-judge",
  MIXED: "mixed",
};

const UNITS: Record<DbUnit, MetricUnit> = {
  PERCENT: "percent",
  SECONDS: "seconds",
  MILLISECONDS: "milliseconds",
  USD: "usd",
  MULTIPLIER: "multiplier",
  COUNT: "count",
};

const HOSTS: Record<DbHost, BenchmarkDataset["host"]> = {
  HUGGING_FACE: "Hugging Face",
  GITHUB: "GitHub",
};

const benchmarkInclude = {
  faqs: { orderBy: { position: "asc" } },
  currentVersion: {
    include: {
      primaryMetric: true,
      tradeoffMetric: true,
      metrics: { orderBy: { position: "asc" } },
      segments: {
        orderBy: { position: "asc" },
        include: { _count: { select: { cases: true } } },
      },
      skills: { orderBy: { position: "asc" } },
      methodology: { orderBy: { position: "asc" } },
      leaderboardEntries: {
        include: {
          model: { include: { provider: true } },
          run: {
            include: { scores: true, segmentScores: true, skillScores: true },
          },
        },
      },
    },
  },
} satisfies Prisma.BenchmarkInclude;

// A benchmark is listed once its current version has a primary metric.
const PUBLISHABLE = {
  currentVersion: { is: { primaryMetricId: { not: null } } },
} satisfies Prisma.BenchmarkWhereInput;

type BenchmarkRow = Prisma.BenchmarkGetPayload<{
  include: typeof benchmarkInclude;
}>;

type CaseCounts = { total: number; heldOut: number };

// Case counts per version, split by whether the case is held out.
async function getCaseCounts(
  versionIds: string[],
): Promise<Map<string, CaseCounts>> {
  const counts = new Map<string, CaseCounts>();
  if (versionIds.length === 0) return counts;

  const groups = await getDb().case.groupBy({
    by: ["versionId", "heldOut"],
    where: { versionId: { in: versionIds } },
    _count: { _all: true },
  });
  for (const group of groups) {
    const current = counts.get(group.versionId) ?? { total: 0, heldOut: 0 };
    current.total += group._count._all;
    if (group.heldOut) current.heldOut += group._count._all;
    counts.set(group.versionId, current);
  }
  return counts;
}

type VersionRow = NonNullable<BenchmarkRow["currentVersion"]>;
type EntryRow = VersionRow["leaderboardEntries"][number];

// Converts one published entry, or returns the reason it is incomplete.
// Every view on the page assumes a value for every metric, segment and
// skill, so a partial entry is left out rather than shown with gaps.
function toEntry(
  row: EntryRow,
  version: VersionRow,
): LeaderboardEntry | string {
  const metricKey = new Map(version.metrics.map((m) => [m.id, m.key]));
  const segmentKey = new Map(version.segments.map((s) => [s.id, s.key]));
  const skillKey = new Map(version.skills.map((s) => [s.id, s.key]));

  const scores: Record<string, number> = {};
  let confidenceInterval: number | undefined;
  for (const score of row.run.scores) {
    const key = metricKey.get(score.metricId);
    if (!key) continue;
    scores[key] = score.value;
    if (score.metricId === version.primaryMetricId) {
      confidenceInterval = score.ciHalfWidth ?? undefined;
    }
  }

  const segmentScores: Record<string, Record<string, number>> = {};
  for (const metric of version.metrics) segmentScores[metric.key] = {};
  for (const score of row.run.segmentScores) {
    const metric = metricKey.get(score.metricId);
    const segment = segmentKey.get(score.segmentId);
    if (metric && segment) segmentScores[metric][segment] = score.value;
  }

  const skillScores: Record<string, number> = {};
  for (const score of row.run.skillScores) {
    const key = skillKey.get(score.skillId);
    if (key) skillScores[key] = score.value;
  }

  const missing = [
    ...version.metrics
      .filter((metric) => scores[metric.key] === undefined)
      .map((metric) => `score:${metric.key}`),
    ...version.metrics.flatMap((metric) =>
      version.segments
        .filter((segment) => segmentScores[metric.key][segment.key] === undefined)
        .map((segment) => `segment:${metric.key}/${segment.key}`),
    ),
    ...version.skills
      .filter((skill) => skillScores[skill.key] === undefined)
      .map((skill) => `skill:${skill.key}`),
  ];
  if (missing.length > 0) return missing.join(", ");

  return {
    model: row.model.name,
    provider: row.model.provider.name,
    scores,
    segmentScores,
    skillScores,
    confidenceInterval,
    openWeights: row.model.openWeights,
    evaluatedAt: (row.run.completedAt ?? row.publishedAt).toISOString(),
  };
}

function toMetric(row: VersionRow["metrics"][number]): BenchmarkMetric {
  return {
    id: row.key,
    label: row.label,
    pickLabel: row.pickLabel ?? undefined,
    unit: UNITS[row.unit],
    higherIsBetter: row.higherIsBetter,
    description: row.description,
    decimals: row.decimals,
  };
}

// Returns undefined for a benchmark with nothing publishable yet: no
// current version, or a current version without a primary metric.
function toBenchmark(
  row: BenchmarkRow,
  caseCounts: Map<string, CaseCounts>,
): Benchmark | undefined {
  const version = row.currentVersion;
  if (!version?.primaryMetric) return undefined;

  const entries: LeaderboardEntry[] = [];
  for (const entryRow of version.leaderboardEntries) {
    const entry = toEntry(entryRow, version);
    if (typeof entry === "string") {
      console.warn(
        `[benchmarks] ${row.slug}: left out ${entryRow.model.name}, missing ${entry}`,
      );
    } else {
      entries.push(entry);
    }
  }

  const counts = caseCounts.get(version.id) ?? { total: 0, heldOut: 0 };
  const latestPublish = version.leaderboardEntries.reduce<Date | undefined>(
    (latest, entry) =>
      !latest || entry.publishedAt > latest ? entry.publishedAt : latest,
    undefined,
  );

  return {
    slug: row.slug,
    name: row.name,
    category: CATEGORIES[row.category],
    tagline: row.tagline,
    summary: row.summary,
    version: version.version,
    updatedAt: (latestPublish ?? version.releasedAt).toISOString(),
    grading: {
      method: GRADING_METHODS[version.gradingMethod],
      summary: version.gradingSummary,
      judgeModel: version.judgeModel ?? undefined,
      runsPerCase: version.runsPerCase,
    },
    dataset: {
      size: counts.total,
      unit: version.datasetUnit,
      source: version.datasetSource,
      license: version.datasetLicense,
      host: HOSTS[version.datasetHost],
      url: version.datasetUrl,
      publicShare:
        counts.total === 0 ? 0 : (counts.total - counts.heldOut) / counts.total,
    },
    metrics: version.metrics.map(toMetric),
    primaryMetricId: version.primaryMetric.key,
    tradeoffMetricId: (version.tradeoffMetric ?? version.primaryMetric).key,
    segmentLabel: version.segmentLabel,
    segments: version.segments.map((segment) => ({
      id: segment.key,
      label: segment.label,
      caseCount: segment._count.cases,
    })),
    skills: version.skills.map((skill) => ({
      id: skill.key,
      label: skill.label,
      description: skill.description,
    })),
    entries,
    methodology: version.methodology.map((step) => ({
      title: step.title,
      body: step.body,
    })),
    runSnippet: row.runSnippet,
    faqs: row.faqs.map((faq) => ({
      question: faq.question,
      answer: faq.answer,
    })),
  };
}

// Page params can arrive still URL-encoded (a name with spaces), and API
// params arrive decoded; both forms resolve the same way.
function normalizeIdentifier(identifier: string): string {
  let value = identifier;
  try {
    value = decodeURIComponent(identifier);
  } catch {
    // Not valid percent-encoding: match the identifier as given.
  }
  return value.trim();
}

// Every benchmark with a publishable current version, in display order.
export async function loadBenchmarksFromDatabase(): Promise<Benchmark[]> {
  const rows = await getDb().benchmark.findMany({
    where: PUBLISHABLE,
    orderBy: [{ position: "asc" }, { name: "asc" }],
    include: benchmarkInclude,
  });
  const caseCounts = await getCaseCounts(
    rows.flatMap((row) => (row.currentVersionId ? [row.currentVersionId] : [])),
  );
  return rows.flatMap((row) => toBenchmark(row, caseCounts) ?? []);
}

// Resolves one benchmark from a URL identifier: its slug first, then its
// exact name. Both columns are unique, and the slug is tried first, so an
// identifier never matches more than one benchmark.
export async function loadBenchmarkFromDatabase(
  identifier: string,
): Promise<Benchmark | undefined> {
  const value = normalizeIdentifier(identifier);
  if (!value) return undefined;

  const db = getDb();
  const row =
    (await db.benchmark.findUnique({
      where: { slug: value.toLowerCase() },
      include: benchmarkInclude,
    })) ??
    (await db.benchmark.findUnique({
      where: { name: value },
      include: benchmarkInclude,
    }));
  if (!row?.currentVersionId) return undefined;

  const caseCounts = await getCaseCounts([row.currentVersionId]);
  return toBenchmark(row, caseCounts);
}

export async function loadBenchmarkSlugsFromDatabase(): Promise<string[]> {
  const rows = await getDb().benchmark.findMany({
    where: PUBLISHABLE,
    orderBy: [{ position: "asc" }, { name: "asc" }],
    select: { slug: true },
  });
  return rows.map((row) => row.slug);
}

// The pages and API read through the three functions below. They serve
// placeholder data for now; swap each body for its `load…FromDatabase`
// counterpart once the database has published results.

export async function getBenchmarks(): Promise<Benchmark[]> {
  return MOCK_BENCHMARKS;
}

// Same resolution rule as `loadBenchmarkFromDatabase`: slug, then exact name.
export async function getBenchmark(
  identifier: string,
): Promise<Benchmark | undefined> {
  const value = normalizeIdentifier(identifier);
  if (!value) return undefined;
  return (
    MOCK_BENCHMARKS.find((benchmark) => benchmark.slug === value.toLowerCase()) ??
    MOCK_BENCHMARKS.find((benchmark) => benchmark.name === value)
  );
}

export async function getBenchmarkSlugs(): Promise<string[]> {
  return MOCK_BENCHMARKS.map((benchmark) => benchmark.slug);
}
