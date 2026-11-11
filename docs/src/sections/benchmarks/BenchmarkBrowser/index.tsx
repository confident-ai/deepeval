"use client";

import { useMemo, useState } from "react";
import { CATEGORY_LABELS } from "@/lib/benchmarks/format";
import type { Benchmark, BenchmarkCategory } from "@/lib/benchmarks/types";
import BenchmarkCards from "../BenchmarkCards";
import FilterPills from "../FilterPills";
import styles from "./BenchmarkBrowser.module.scss";

type BenchmarkBrowserProps = {
  benchmarks: Benchmark[];
};

type Filter = BenchmarkCategory | "all";

const BenchmarkBrowser: React.FC<BenchmarkBrowserProps> = ({ benchmarks }) => {
  const [filter, setFilter] = useState<Filter>("all");

  // Only categories that have at least one benchmark, in first-seen order.
  const tabs = useMemo(() => {
    const counts = new Map<BenchmarkCategory, number>();
    for (const benchmark of benchmarks) {
      counts.set(benchmark.category, (counts.get(benchmark.category) ?? 0) + 1);
    }
    return [
      { id: "all" as Filter, label: "All", count: benchmarks.length },
      ...[...counts].map(([category, count]) => ({
        id: category as Filter,
        label: CATEGORY_LABELS[category],
        count,
      })),
    ];
  }, [benchmarks]);

  const shown =
    filter === "all"
      ? benchmarks
      : benchmarks.filter((benchmark) => benchmark.category === filter);

  return (
    <div className={styles.wrap}>
      <FilterPills
        label="Benchmark type"
        value={filter}
        onChange={setFilter}
        options={tabs}
      />
      {shown.length > 0 ? (
        <BenchmarkCards benchmarks={shown} />
      ) : (
        <p className={styles.empty}>No benchmarks are published yet.</p>
      )}
    </div>
  );
};

export default BenchmarkBrowser;
