"use client";

import { useMemo, useState } from "react";
import * as Popover from "@radix-ui/react-popover";
import { ChartScatter, Check, ChevronDown, Search, Table2 } from "lucide-react";
import {
  formatCount,
  sortEntries,
  type TradeoffPoint,
} from "@/lib/benchmarks/format";
import type {
  BenchmarkMetric,
  BenchmarkSegment,
  LeaderboardEntry,
  RankedEntry,
} from "@/lib/benchmarks/types";
import LeaderboardTable from "../LeaderboardTable";
import TradeoffChart from "../TradeoffChart";
import ViewToggle from "../ViewToggle";
import styles from "./LeaderboardExplorer.module.scss";

type LeaderboardExplorerProps = {
  entries: LeaderboardEntry[];
  metrics: BenchmarkMetric[];
  primaryMetricId: string;
  tradeoffMetricId: string;
  segmentLabel: string;
  segments: BenchmarkSegment[];
  totalCases: number;
  caseUnit: string;
};

type View = "table" | "chart";

const OVERALL = "overall";

// Segments beyond this many move into a searchable menu, so the row stays
// one line however many a benchmark has.
const MAX_INLINE = 6;

const LeaderboardExplorer: React.FC<LeaderboardExplorerProps> = ({
  entries,
  metrics,
  primaryMetricId,
  tradeoffMetricId,
  segmentLabel,
  segments,
  totalCases,
  caseUnit,
}) => {
  const [segmentId, setSegmentId] = useState(OVERALL);
  const [view, setView] = useState<View>("table");
  const [menuOpen, setMenuOpen] = useState(false);
  const [query, setQuery] = useState("");

  const primary = metrics.find((metric) => metric.id === primaryMetricId)!;
  const tradeoff = metrics.find((metric) => metric.id === tradeoffMetricId)!;
  const segment = segments.find((item) => item.id === segmentId);

  // With a segment selected, the primary column shows that segment's score
  // and ranks are recomputed from it.
  const ranked = useMemo<RankedEntry[]>(() => {
    const scoped = entries.map((entry) =>
      segment
        ? {
            ...entry,
            scores: {
              ...entry.scores,
              [primary.id]: entry.segmentScores[primary.id][segment.id],
            },
            confidenceInterval: undefined,
          }
        : entry,
    );
    return sortEntries(scoped, primary).map((entry, index) => ({
      ...entry,
      rank: index + 1,
    }));
  }, [entries, primary, segment]);

  const points = useMemo<TradeoffPoint[]>(
    () =>
      ranked.map((entry) => ({
        model: entry.model,
        provider: entry.provider,
        rank: entry.rank,
        x: entry.scores[tradeoff.id],
        y: entry.scores[primary.id],
      })),
    [ranked, tradeoff.id, primary.id],
  );

  const overflowing = segments.length > MAX_INLINE;
  const inline = overflowing ? segments.slice(0, MAX_INLINE - 1) : segments;
  const overflow = overflowing ? segments.slice(MAX_INLINE - 1) : [];
  const tabs = [
    { id: OVERALL, label: "Overall", caseCount: totalCases },
    ...inline,
  ];
  const overflowSelected = overflow.find((item) => item.id === segmentId);
  const matches = overflow.filter((item) =>
    item.label.toLowerCase().includes(query.trim().toLowerCase()),
  );

  const pick = (id: string) => {
    setSegmentId(id);
    setMenuOpen(false);
    setQuery("");
  };

  return (
    <div className={styles.wrap}>
      <div className={styles.controls}>
        <div className={styles.filter}>
          <div
            className={styles.tabs}
            role="radiogroup"
            aria-label={segmentLabel}
          >
            {tabs.map((tab) => (
              <button
                key={tab.id}
                type="button"
                role="radio"
                aria-checked={tab.id === segmentId}
                className={styles.tab}
                onClick={() => setSegmentId(tab.id)}
              >
                {tab.label}
                <span className={styles.count}>{formatCount(tab.caseCount)}</span>
              </button>
            ))}
          </div>

          {overflowing && (
            <Popover.Root open={menuOpen} onOpenChange={setMenuOpen}>
              <Popover.Trigger
                className={styles.more}
                data-active={Boolean(overflowSelected)}
              >
                {overflowSelected ? (
                  <>
                    {overflowSelected.label}
                    <span className={styles.count}>
                      {formatCount(overflowSelected.caseCount)}
                    </span>
                  </>
                ) : (
                  `${overflow.length} more`
                )}
                <ChevronDown aria-hidden />
              </Popover.Trigger>
              <Popover.Portal>
                <Popover.Content
                  align="start"
                  sideOffset={6}
                  className={styles.menu}
                >
                  <label className={styles.search}>
                    <Search aria-hidden />
                    <input
                      type="search"
                      value={query}
                      onChange={(event) => setQuery(event.target.value)}
                      placeholder={`Search ${segmentLabel.toLowerCase()}`}
                      aria-label={`Search ${segmentLabel.toLowerCase()}`}
                      autoFocus
                    />
                  </label>
                  <ul className={styles.menuList}>
                    {matches.map((item) => (
                      <li key={item.id}>
                        <button
                          type="button"
                          className={styles.menuItem}
                          aria-pressed={item.id === segmentId}
                          onClick={() => pick(item.id)}
                        >
                          <Check aria-hidden />
                          <span>{item.label}</span>
                          <span className={styles.count}>
                            {formatCount(item.caseCount)}
                          </span>
                        </button>
                      </li>
                    ))}
                    {matches.length === 0 && (
                      <li className={styles.empty}>No matches</li>
                    )}
                  </ul>
                </Popover.Content>
              </Popover.Portal>
            </Popover.Root>
          )}
        </div>

        <ViewToggle<View>
          value={view}
          onChange={setView}
          options={[
            { id: "table", label: "Table", icon: <Table2 aria-hidden /> },
            {
              id: "chart",
              label: "Trade-off",
              icon: <ChartScatter aria-hidden />,
            },
          ]}
        />
      </div>

      <p className={styles.scope}>
        {segment
          ? `${primary.label} on ${segment.label} only (${formatCount(segment.caseCount)} ${caseUnit}). Other columns are overall.`
          : `All ${formatCount(totalCases)} ${caseUnit}.`}
      </p>

      {view === "table" ? (
        <LeaderboardTable
          key={segmentId}
          entries={ranked}
          metrics={metrics}
          primaryMetricId={primaryMetricId}
        />
      ) : (
        <TradeoffChart points={points} xMetric={tradeoff} yMetric={primary} />
      )}
    </div>
  );
};

export default LeaderboardExplorer;
