"use client";

import { useState } from "react";
import type {
  BenchmarkJourney,
  BenchmarkSegment,
  SkillFamily,
} from "@/lib/benchmarks/types";
import FilterPills, { type PillOption } from "../FilterPills";
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
  const [familyId, setFamilyId] = useState(ALL);
  const [skillId, setSkillId] = useState(ALL);
  const [activeId, setActiveId] = useState(journeys[0].id);

  const testing = (skills: string[]) =>
    journeys.filter((journey) =>
      skills.some((skill) => journey.skills.includes(skill)),
    );
  const skillsOf = (family: SkillFamily) =>
    family.skills.map((skill) => skill.id);

  const family = families.find((candidate) => candidate.id === familyId);
  const familyPills: PillOption<string>[] = [
    { id: ALL, label: "All", count: journeys.length },
    ...families.map((candidate) => ({
      id: candidate.id,
      label: candidate.label,
      count: testing(skillsOf(candidate)).length,
    })),
  ];
  const skillPills: PillOption<string>[] = family
    ? [
        { id: ALL, label: "Any", count: testing(skillsOf(family)).length },
        ...family.skills.map((skill) => ({
          id: skill.id,
          label: skill.label,
          count: testing([skill.id]).length,
        })),
      ]
    : [];

  const topics = new Map(
    segments.map((segment) => [segment.id, segment.label]),
  );
  const shown = !family
    ? journeys
    : testing(skillId === ALL ? skillsOf(family) : [skillId]);
  const active = shown.find((journey) => journey.id === activeId) ?? shown[0];

  return (
    <div className={styles.explorer}>
      <div className={styles.filters}>
        <FilterPills
          options={familyPills}
          value={familyId}
          onChange={(id) => {
            setFamilyId(id);
            setSkillId(ALL);
          }}
          label="What is tested"
        />
        {family && (
          <FilterPills
            options={skillPills}
            value={skillId}
            onChange={setSkillId}
            label="Skill"
          />
        )}
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
