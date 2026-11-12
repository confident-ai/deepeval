"use client";

import { useState } from "react";
import type {
  BenchmarkJourney,
  BenchmarkSegment,
  SkillFamily,
} from "@/lib/benchmarks/types";
import FilterMenu, { type FilterGroup, type FilterOption } from "../FilterMenu";
import JourneyGraph from "../JourneyGraph";
import styles from "./JourneyExplorer.module.scss";

type JourneyExplorerProps = {
  journeys: BenchmarkJourney[];
  segments: BenchmarkSegment[];
  families: SkillFamily[];
};

const LINES = ["Phone line", "Weak mobile line", "Speakerphone", "Landline"];
const AUDIO_PROBLEMS = [
  "Audio cuts out",
  "Garbled audio",
  "Echo",
  "Delay on the line",
  "Static",
];

const JourneyExplorer: React.FC<JourneyExplorerProps> = ({
  journeys,
  segments,
  families,
}) => {
  const [skills, setSkills] = useState<string[]>([]);
  const [conditions, setConditions] = useState<string[]>([]);
  const [activeId, setActiveId] = useState(journeys[0].id);

  const accentOf = (journey: BenchmarkJourney) =>
    journey.caller.split(" · ")[0];
  const has = (journey: BenchmarkJourney, id: string) =>
    journey.skills.includes(id) ||
    journey.conditions.includes(id) ||
    accentOf(journey) === id;
  const option = (id: string, label: string): FilterOption => ({
    id,
    label,
    count: journeys.filter((journey) => has(journey, id)).length,
  });
  const named = (ids: string[]) =>
    ids
      .map((id) => option(id, id))
      .filter((entry) => entry.count > 0)
      .sort((a, b) => b.count - a.count);

  const skillGroups: FilterGroup[] = families.map((family) => ({
    id: family.id,
    label: family.label,
    options: family.skills.map((skill) => option(skill.id, skill.label)),
  }));
  const everyCondition = [
    ...new Set(journeys.flatMap((journey) => journey.conditions)),
  ];
  const conditionGroups: FilterGroup[] = [
    { id: "line", label: "Line", options: named(LINES) },
    { id: "audio", label: "Audio problems", options: named(AUDIO_PROBLEMS) },
    {
      id: "background",
      label: "Background",
      options: named(
        everyCondition.filter(
          (name) => !LINES.includes(name) && !AUDIO_PROBLEMS.includes(name),
        ),
      ),
    },
    {
      id: "accent",
      label: "Accent",
      options: named([...new Set(journeys.map(accentOf))]),
    },
  ];

  // Ticks in one group widen the search; each group then narrows it.
  const passes = (
    journey: BenchmarkJourney,
    groups: FilterGroup[],
    ticked: string[],
  ) =>
    groups.every((group) => {
      const wanted = group.options
        .map((entry) => entry.id)
        .filter((id) => ticked.includes(id));
      return wanted.length === 0 || wanted.some((id) => has(journey, id));
    });

  const topics = new Map(
    segments.map((segment) => [segment.id, segment.label]),
  );
  const matching = journeys.filter(
    (journey) =>
      passes(journey, skillGroups, skills) &&
      passes(journey, conditionGroups, conditions),
  );
  const shown = matching.length > 0 ? matching : journeys;
  const active = shown.find((journey) => journey.id === activeId) ?? shown[0];

  return (
    <div className={styles.explorer}>
      <div className={styles.filters}>
        <FilterMenu
          label="Skills tested"
          groups={skillGroups}
          selected={skills}
          onChange={setSkills}
        />
        <FilterMenu
          label="Call conditions"
          groups={conditionGroups}
          selected={conditions}
          onChange={setConditions}
        />
        <span className={styles.found}>
          {matching.length === 0
            ? "No journey matches, showing all"
            : `${matching.length} of ${journeys.length} journeys`}
        </span>
      </div>

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
              <div className={styles.about}>
                <h3>{active.title}</h3>
                <p>{active.summary}</p>
              </div>
              <dl className={styles.tags}>
                <dt>Skills tested</dt>
                <dd className={styles.tested}>
                  {active.skills.map((skill) => (
                    <span key={skill}>{skill.replaceAll("_", " ")}</span>
                  ))}
                </dd>
                <dt>Call conditions</dt>
                <dd className={styles.conditions}>
                  <span>{active.caller.split(" · ")[0]} caller</span>
                  {active.conditions.map((condition) => (
                    <span key={condition}>{condition}</span>
                  ))}
                </dd>
              </dl>
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
