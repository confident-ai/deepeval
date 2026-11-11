"use client";

import { useState } from "react";
import type { BenchmarkSkill, LeaderboardEntry } from "@/lib/benchmarks/types";
import ModelName from "../ModelName";
import { RankBadge } from "../RankTrophy";
import styles from "./SkillBreakdown.module.scss";

type SkillBreakdownProps = {
  skills: BenchmarkSkill[];
  entries: LeaderboardEntry[];
};

const SkillBreakdown: React.FC<SkillBreakdownProps> = ({ skills, entries }) => {
  const [activeId, setActiveId] = useState(skills[0].id);
  const active = skills.find((skill) => skill.id === activeId) ?? skills[0];

  const ranked = [...entries].sort(
    (a, b) => b.skillScores[active.id] - a.skillScores[active.id],
  );

  return (
    <div className={styles.wrap}>
      <div className={styles.tabs} role="tablist" aria-label="Skills">
        {skills.map((skill) => {
          const leader = [...entries].sort(
            (a, b) => b.skillScores[skill.id] - a.skillScores[skill.id],
          )[0];
          return (
            <button
              key={skill.id}
              type="button"
              role="tab"
              aria-selected={skill.id === active.id}
              className={styles.tab}
              onClick={() => setActiveId(skill.id)}
            >
              <span className={styles.tabLabel}>{skill.label}</span>
              <span className={styles.tabLeader}>{leader.model}</span>
            </button>
          );
        })}
      </div>

      <div className={styles.panel} role="tabpanel">
        <p className={styles.description}>{active.description}</p>
        <div className={styles.listHead}>
          <span>Ranked on this skill</span>
          <span>Score out of 100</span>
        </div>
        <ol className={styles.list}>
          {ranked.map((entry, index) => {
            const score = entry.skillScores[active.id];
            return (
              <li key={entry.model} className={styles.row} data-top={index === 0}>
                <span className={styles.rank}>
                  <RankBadge rank={index + 1} />
                </span>
                <span className={styles.model}>
                  <ModelName model={entry.model} provider={entry.provider} />
                </span>
                <span className={styles.track} aria-hidden>
                  <span className={styles.fill} style={{ width: `${score}%` }} />
                </span>
                <span className={styles.score}>{score}</span>
              </li>
            );
          })}
        </ol>
      </div>
    </div>
  );
};

export default SkillBreakdown;
