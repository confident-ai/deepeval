"use client";

import { useState } from "react";
import type {
  BenchmarkJourney,
  BenchmarkSegment,
  SkillFamily,
} from "@/lib/benchmarks/types";
import JourneyGraph from "../JourneyGraph";
import styles from "./JourneyExplorer.module.scss";

type JourneyExplorerProps = {
  journeys: BenchmarkJourney[];
  segments: BenchmarkSegment[];
  families: SkillFamily[];
};

const ALL = "all";

const JourneyExplorer: React.FC<JourneyExplorerProps> = ({
  journeys,
  segments,
  families,
}) => {
  const [skill, setSkill] = useState(ALL);
  const [condition, setCondition] = useState(ALL);
  const [activeId, setActiveId] = useState(journeys[0].id);

  const matches = (
    journey: BenchmarkJourney,
    wantedSkill: string,
    wantedCondition: string,
  ) =>
    (wantedSkill === ALL || journey.skills.includes(wantedSkill)) &&
    (wantedCondition === ALL || journey.conditions.includes(wantedCondition));
  const count = (wantedSkill: string, wantedCondition: string) =>
    journeys.filter((journey) => matches(journey, wantedSkill, wantedCondition))
      .length;

  const conditions = [
    ...new Set(journeys.flatMap((journey) => journey.conditions)),
  ]
    .map((name) => ({ name, total: count(ALL, name) }))
    .sort((a, b) => b.total - a.total);

  const topics = new Map(
    segments.map((segment) => [segment.id, segment.label]),
  );
  const matching = journeys.filter((journey) =>
    matches(journey, skill, condition),
  );
  const shown = matching.length > 0 ? matching : journeys;
  const active = shown.find((journey) => journey.id === activeId) ?? shown[0];

  return (
    <div className={styles.explorer}>
      <div className={styles.filters}>
        <label>
          Skills tested
          <select
            value={skill}
            onChange={(event) => setSkill(event.target.value)}
          >
            <option value={ALL}>All skills ({count(ALL, condition)})</option>
            {families.map((family) => (
              <optgroup key={family.id} label={family.label}>
                {family.skills.map((option) => (
                  <option key={option.id} value={option.id}>
                    {option.label} ({count(option.id, condition)})
                  </option>
                ))}
              </optgroup>
            ))}
          </select>
        </label>
        <label>
          Call conditions
          <select
            value={condition}
            onChange={(event) => setCondition(event.target.value)}
          >
            <option value={ALL}>All conditions ({count(skill, ALL)})</option>
            {conditions.map(({ name }) => (
              <option key={name} value={name}>
                {name} ({count(skill, name)})
              </option>
            ))}
          </select>
        </label>
        <span className={styles.found}>
          {matching.length} of {journeys.length} journeys
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
