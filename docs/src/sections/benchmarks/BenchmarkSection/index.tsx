import type { ReactNode } from "react";
import { ArrowUpRight } from "lucide-react";
import { ServerCodeBlock } from "fumadocs-ui/components/codeblock.rsc";
import { cn } from "@/lib/cn";
import SectionLabel from "@/src/components/SectionLabel";
import { externalRelForOutboundHref } from "@/src/utils/outbound-link-rel";
import {
  GRADING_LABELS,
  formatCount,
  type Benchmark,
  type MethodologyStep,
} from "@/lib/benchmarks";
import styles from "./BenchmarkSection.module.scss";

type BenchmarkSectionProps = {
  id?: string;
  label: string;
  title: string;
  description?: ReactNode;
  fullWidth?: boolean;
  children?: ReactNode;
};

export const BenchmarkSection: React.FC<BenchmarkSectionProps> = ({
  id,
  label,
  title,
  description,
  fullWidth = false,
  children,
}) => {
  return (
    <section
      id={id}
      className={cn(styles.section, fullWidth && styles.fullWidth)}
    >
      <div className={cn("prose", styles.prose, styles.intro)}>
        <SectionLabel>{label}</SectionLabel>
        <h2>{title}</h2>
        {description && <p>{description}</p>}
      </div>
      {children && <div className={styles.body}>{children}</div>}
    </section>
  );
};

// Gives embedded docs components (code blocks, FAQs) the same typography
// they have inside a docs page.
export const Prose: React.FC<{ children: ReactNode }> = ({ children }) => (
  <div className={cn("prose", styles.prose)}>{children}</div>
);

type Fact = {
  label: string;
  value: ReactNode;
};

const FactList: React.FC<{ facts: Fact[] }> = ({ facts }) => (
  <dl className={styles.factList}>
    {facts.map((fact) => (
      <div key={fact.label} className={styles.factRow}>
        <dt>{fact.label}</dt>
        <dd>{fact.value}</dd>
      </div>
    ))}
  </dl>
);

export const StepGrid: React.FC<{ steps: MethodologyStep[] }> = ({ steps }) => (
  <ol className={styles.steps}>
    {steps.map((step, index) => (
      <li key={step.title} className={styles.step}>
        <span className={styles.stepIndex}>
          {String(index + 1).padStart(2, "0")}
        </span>
        <h3 className={styles.stepTitle}>{step.title}</h3>
        <p className={styles.stepBody}>{step.body}</p>
      </li>
    ))}
  </ol>
);

export const MethodologySection: React.FC<{ benchmark: Benchmark }> = ({
  benchmark,
}) => {
  const { grading } = benchmark;
  const facts: Fact[] = [
    { label: "Grading", value: GRADING_LABELS[grading.method] },
    { label: "Judge model", value: grading.judgeModel ?? "None" },
    { label: "Runs per case", value: String(grading.runsPerCase) },
    { label: "Human graders", value: "None" },
  ];

  return (
    <BenchmarkSection
      id="methodology"
      label="Methodology"
      title="How it is graded"
      description={grading.summary}
    >
      <FactList facts={facts} />
      {benchmark.methodology.length > 0 && (
        <StepGrid steps={benchmark.methodology} />
      )}
    </BenchmarkSection>
  );
};

export const DatasetSection: React.FC<{
  benchmark: Benchmark;
  // The test cases themselves, for benchmarks that can show them.
  children?: ReactNode;
}> = ({ benchmark, children }) => {
  const { dataset } = benchmark;
  const publicPercent = Math.round(dataset.publicShare * 100);
  const facts: Fact[] = [
    {
      label: "Size",
      value: `${formatCount(dataset.size)} ${dataset.unit}`,
    },
    { label: "Source", value: dataset.source },
    { label: "License", value: dataset.license },
    {
      label: "Hosted on",
      value: (
        <a
          href={dataset.url}
          target="_blank"
          rel={externalRelForOutboundHref(dataset.url)}
          className={styles.link}
        >
          {dataset.host}
          <ArrowUpRight aria-hidden />
        </a>
      ),
    },
    {
      label: "Public",
      value: (
        <span className={styles.share}>
          {publicPercent}%
          <span className={styles.shareTrack} aria-hidden>
            <span
              className={styles.shareFill}
              style={{ width: `${publicPercent}%` }}
            />
          </span>
        </span>
      ),
    },
  ];

  return (
    <BenchmarkSection
      id="dataset"
      label="Dataset"
      title="What the models are tested on"
      description={
        publicPercent === 100
          ? "The full test set is public, so every row on the leaderboard can be reproduced."
          : `${publicPercent}% of the test set is public. The rest is held out to check for training-data overlap.`
      }
      fullWidth={Boolean(children)}
    >
      <FactList facts={facts} />
      {children}
    </BenchmarkSection>
  );
};

export const RunSection: React.FC<{ benchmark: Benchmark }> = ({
  benchmark,
}) => {
  return (
    <BenchmarkSection
      id="run"
      label="Reproduce"
      title="Run it with DeepEval"
      description="Point the benchmark at any model you can call from Python. It downloads the dataset, runs every case and applies the same grading as the leaderboard."
    >
      <Prose>
        <ServerCodeBlock lang="python" code={benchmark.runSnippet} />
      </Prose>
    </BenchmarkSection>
  );
};
