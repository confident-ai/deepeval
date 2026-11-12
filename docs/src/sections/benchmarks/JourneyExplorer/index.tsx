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

type Grouping = "industry" | "skill";

const ALL = "all";

const JourneyExplorer: React.FC<JourneyExplorerProps> = ({
  journeys,
  segments,
  families,
}) => {
  const [grouping, setGrouping] = useState<Grouping>("industry");
  const [segmentId, setSegmentId] = useState(ALL);
  const [familyId, setFamilyId] = useState(families[0]?.id ?? ALL);
  const [skillId, setSkillId] = useState(ALL);
  const [activeId, setActiveId] = useState(journeys[0].id);

  const tests = (journey: BenchmarkJourney, skills: string[]) =>
    skills.some((skill) => journey.skills.includes(skill));
  const count = (matching: BenchmarkJourney[]) => matching.length;

  const industries: PillOption<string>[] = [
    { id: ALL, label: "All", count: journeys.length },
    ...segments.map((segment) => ({
      id: segment.id,
      label: segment.label,
      count: count(journeys.filter((journey) => journey.segmentId === segment.id)),
    })),
  ];
  const family = families.find((candidate) => candidate.id === familyId);
  const familySkills = (family?.skills ?? []).map((skill) => skill.id);
  const familyPills: PillOption<string>[] = families.map((candidate) => ({
    id: candidate.id,
    label: candidate.label,
    count: count(
      journeys.filter((journey) =>
        tests(journey, candidate.skills.map((skill) => skill.id)),
      ),
    ),
  }));
  const skillPills: PillOption<string>[] = [
    {
      id: ALL,
      label: "Any",
      count: count(journeys.filter((journey) => tests(journey, familySkills))),
    },
    ...(family?.skills ?? []).map((skill) => ({
      id: skill.id,
      label: skill.label,
      count: count(journeys.filter((journey) => tests(journey, [skill.id]))),
    })),
  ];

  const topics = new Map(
    segments.map((segment) => [segment.id, segment.label]),
  );
  const matching =
    grouping === "industry"
      ? journeys.filter(
          (journey) => segmentId === ALL || journey.segmentId === segmentId,
        )
      : journeys.filter((journey) =>
          tests(journey, skillId === ALL ? familySkills : [skillId]),
        );
  const shown = matching.length > 0 ? matching : journeys;
  const active = shown.find((journey) => journey.id === activeId) ?? shown[0];

  return (
    <div className={styles.explorer}>
      <div className={styles.filters}>
        <FilterPills
          options={[
            { id: "industry", label: "By industry" },
            { id: "skill", label: "By what is tested" },
          ]}
          value={grouping}
          onChange={setGrouping}
          label="Group journeys"
        />
        {grouping === "industry" ? (
          <FilterPills
            options={industries}
            value={segmentId}
            onChange={setSegmentId}
            label="Industry"
          />
        ) : (
          <>
            <FilterPills
              options={familyPills}
              value={familyId}
              onChange={(id) => {
                setFamilyId(id);
                setSkillId(ALL);
              }}
              label="Skill family"
            />
            <FilterPills
              options={skillPills}
              value={skillId}
              onChange={setSkillId}
              label="Skill"
            />
          </>
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
              <h3>{active.title}</h3>
              <p>{active.summary}</p>
              <div className={styles.tested}>
                {active.skills.map((skill) => (
                  <span key={skill}>{skill.replaceAll("_", " ")}</span>
                ))}
              </div>
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
