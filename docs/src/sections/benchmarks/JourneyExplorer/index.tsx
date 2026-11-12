"use client";

import { useState } from "react";
import type {
  BenchmarkJourney,
  BenchmarkSegment,
} from "@/lib/benchmarks/types";
import FilterPills, { type PillOption } from "../FilterPills";
import JourneyGraph from "../JourneyGraph";
import styles from "./JourneyExplorer.module.scss";

type JourneyExplorerProps = {
  journeys: BenchmarkJourney[];
  segments: BenchmarkSegment[];
};

const ALL = "all";

const JourneyExplorer: React.FC<JourneyExplorerProps> = ({
  journeys,
  segments,
}) => {
  const [segmentId, setSegmentId] = useState(ALL);
  const [activeId, setActiveId] = useState(journeys[0].id);

  const options: PillOption<string>[] = [
    { id: ALL, label: "All", count: journeys.length },
    ...segments.map((segment) => ({
      id: segment.id,
      label: segment.label,
      count: journeys.filter((journey) => journey.segmentId === segment.id)
        .length,
    })),
  ];
  const topics = new Map(
    segments.map((segment) => [segment.id, segment.label]),
  );
  const shown =
    segmentId === ALL
      ? journeys
      : journeys.filter((journey) => journey.segmentId === segmentId);
  const active = shown.find((journey) => journey.id === activeId) ?? shown[0];

  return (
    <div className={styles.explorer}>
      <FilterPills
        options={options}
        value={segmentId}
        onChange={setSegmentId}
        label="Industry"
      />

      <div className={styles.wrap}>
        <div className={styles.top}>
          <div className={styles.rail}>
            <div className={styles.tabs} role="tablist" aria-label="Journeys">
          {shown.map((journey) => (
            <button
              key={journey.id}
              type="button"
              role="tab"
              aria-selected={journey.id === active.id}
              className={styles.tab}
              onClick={() => setActiveId(journey.id)}
            >
              <span className={styles.number}>
                {String(journeys.indexOf(journey) + 1).padStart(2, "0")}
              </span>
              <span className={styles.tabBody}>
                <span className={styles.tabLabel}>{journey.name}</span>
                <span className={styles.blurb}>{journey.blurb}</span>
                <span className={styles.topic}>
                  {topics.get(journey.segmentId)}
                </span>
              </span>
            </button>
          ))}
            </div>
          </div>

          <div className={styles.main}>
            <div className={styles.scenario}>
              <h3>{active.title}</h3>
              <p>{active.summary}</p>
              <div className={styles.conditions}>
                <span>{active.caller.split(" · ")[0]} caller</span>
                {active.conditions.map((condition) => (
                  <span key={condition}>{condition}</span>
                ))}
              </div>
            </div>
            <JourneyGraph
              key={active.id}
              graph={active.graph}
              lines={active.lines}
            />
          </div>
        </div>
      </div>
    </div>
  );
};

export default JourneyExplorer;
