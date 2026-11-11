import type { Metadata } from "next";
import { notFound } from "next/navigation";
import { CATEGORY_LABELS, getMetric } from "@/lib/benchmarks";
import { getBenchmark } from "@/lib/benchmarks/repository";
import FAQs from "@/src/components/FAQ";
import BenchmarkHeader from "@/src/sections/benchmarks/BenchmarkHeader";
import BenchmarkLayout, {
  type SectionLink,
} from "@/src/sections/benchmarks/BenchmarkLayout";
import BestPicks from "@/src/sections/benchmarks/BestPicks";
import LeaderboardExplorer from "@/src/sections/benchmarks/LeaderboardExplorer";
import RadarCompare from "@/src/sections/benchmarks/RadarCompare";
import SegmentSection from "@/src/sections/benchmarks/SegmentSection";
import SkillBreakdown from "@/src/sections/benchmarks/SkillBreakdown";
import {
  BenchmarkSection,
  DatasetSection,
  MethodologySection,
  Prose,
  RunSection,
} from "@/src/sections/benchmarks/BenchmarkSection";

type PageProps = {
  params: Promise<{ slug: string }>;
};

// Results are read from the database on each request.
export const dynamic = "force-dynamic";

export async function generateMetadata({
  params,
}: PageProps): Promise<Metadata> {
  const { slug } = await params;
  const benchmark = await getBenchmark(slug);
  if (!benchmark) return {};

  return {
    title: `${benchmark.name} Leaderboard`,
    description: `${benchmark.name} leaderboard for ${CATEGORY_LABELS[benchmark.category].toLowerCase()}: ${benchmark.entries.length} models compared, with the grading method, dataset and the DeepEval command to reproduce every row.`,
    alternates: { canonical: `/benchmarks/${benchmark.slug}` },
  };
}

export default async function BenchmarkPage({ params }: PageProps) {
  const { slug } = await params;
  const benchmark = await getBenchmark(slug);
  if (!benchmark) notFound();

  const primary = getMetric(benchmark.metrics, benchmark.primaryMetricId);
  const segmentLabel = benchmark.segmentLabel.toLowerCase();

  const hasResults = benchmark.entries.length > 0;
  const hasSegments = hasResults && benchmark.segments.length > 0;
  const hasSkills = hasResults && benchmark.skills.length > 0;
  const hasFaqs = benchmark.faqs.length > 0;

  const sections: SectionLink[] = [
    { id: "leaderboard", label: "Leaderboard" },
    ...(hasResults ? [{ id: "best-picks", label: "Best picks" }] : []),
    ...(hasSegments ? [{ id: "segments", label: `By ${segmentLabel}` }] : []),
    ...(hasSkills
      ? [
          { id: "skills", label: "Skill by skill" },
          { id: "compare", label: "Compare" },
        ]
      : []),
    { id: "methodology", label: "Methodology" },
    { id: "dataset", label: "Dataset" },
    { id: "run", label: "Reproduce" },
    ...(hasFaqs ? [{ id: "faq", label: "FAQ" }] : []),
  ];

  return (
    <>
      <BenchmarkHeader benchmark={benchmark} />

      <BenchmarkLayout sections={sections}>
        {hasResults ? (
          <BenchmarkSection
            id="leaderboard"
            label="Leaderboard"
            title={`Ranked by ${primary.label.toLowerCase()}`}
            description={`${primary.description} ${primary.higherIsBetter ? "Higher" : "Lower"} is better. Filter by ${segmentLabel}, sort by any column, or switch to the trade-off view.`}
            fullWidth
          >
            <LeaderboardExplorer
              entries={benchmark.entries}
              metrics={benchmark.metrics}
              primaryMetricId={benchmark.primaryMetricId}
              tradeoffMetricId={benchmark.tradeoffMetricId}
              segmentLabel={benchmark.segmentLabel}
              segments={benchmark.segments}
              totalCases={benchmark.dataset.size}
              caseUnit={benchmark.dataset.unit}
            />
          </BenchmarkSection>
        ) : (
          <BenchmarkSection
            id="leaderboard"
            label="Leaderboard"
            title="No published results yet"
            description="Models are added to this leaderboard once their runs are complete and checked. The methodology and dataset below are final for this version."
          />
        )}

        {hasResults && (
          <BenchmarkSection
            id="best-picks"
            label="Best picks"
            title="The leader on each measure"
            description="The top-ranked model is not always the one to choose. Each card ranks the same models on a different column."
            fullWidth
          >
            <BestPicks benchmark={benchmark} />
          </BenchmarkSection>
        )}

        {hasSegments && (
          <BenchmarkSection
            id="segments"
            label={`By ${segmentLabel}`}
            title={`Results by ${segmentLabel}`}
            description="An overall score hides where a model is uneven. Pick a metric to see how each model does on every slice of the test set, as bars or as a heatmap."
            fullWidth
          >
            <SegmentSection benchmark={benchmark} />
          </BenchmarkSection>
        )}

        {hasSkills && (
          <>
            <BenchmarkSection
              id="skills"
              label="Skills"
              title="Who does each skill best"
              description="Every test case is tagged with the skill it exercises. Scores are out of 100, and the leader on one skill is often not the leader on the next."
              fullWidth
            >
              <SkillBreakdown
                skills={benchmark.skills}
                entries={benchmark.entries}
              />
            </BenchmarkSection>

            <BenchmarkSection
              id="compare"
              label="Compare"
              title="Two models, side by side"
              description="Pick any two models to see where one leads and the other trails. A bigger shape is a stronger model."
              fullWidth
            >
              <RadarCompare
                skills={benchmark.skills}
                entries={benchmark.entries}
              />
            </BenchmarkSection>
          </>
        )}

        <MethodologySection benchmark={benchmark} />
        <DatasetSection benchmark={benchmark} />
        <RunSection benchmark={benchmark} />

        {hasFaqs && (
          <BenchmarkSection id="faq" label="FAQ" title="Common questions">
            <Prose>
              <FAQs qas={benchmark.faqs} />
            </Prose>
          </BenchmarkSection>
        )}
      </BenchmarkLayout>
    </>
  );
}
