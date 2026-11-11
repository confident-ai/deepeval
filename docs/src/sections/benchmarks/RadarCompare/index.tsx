"use client";

import { useState } from "react";
import type { BenchmarkSkill, LeaderboardEntry } from "@/lib/benchmarks/types";
import ModelName from "../ModelName";
import styles from "./RadarCompare.module.scss";

type RadarCompareProps = {
  skills: BenchmarkSkill[];
  entries: LeaderboardEntry[];
};

const SIZE = 420;
const CENTER = SIZE / 2;
const RADIUS = 130;
const RINGS = [25, 50, 75, 100];
const MAX_SELECTED = 2;

function describe(entry: LeaderboardEntry, skills: BenchmarkSkill[]): string {
  const sorted = [...skills].sort(
    (a, b) => entry.skillScores[b.id] - entry.skillScores[a.id],
  );
  const best = sorted[0];
  const worst = sorted[sorted.length - 1];
  return `strongest at ${best.label.toLowerCase()} (${entry.skillScores[best.id]}), weakest at ${worst.label.toLowerCase()} (${entry.skillScores[worst.id]}).`;
}

const RadarCompare: React.FC<RadarCompareProps> = ({ skills, entries }) => {
  // Fixed slots, so a model keeps its colour while the other one changes.
  const [slots, setSlots] = useState<(string | null)[]>(
    entries.slice(0, MAX_SELECTED).map((entry) => entry.model),
  );
  const [nextSlot, setNextSlot] = useState(0);

  const toggle = (model: string) => {
    const index = slots.indexOf(model);
    if (index !== -1) {
      if (slots.filter(Boolean).length > 1) {
        setSlots(slots.map((value, i) => (i === index ? null : value)));
      }
      return;
    }
    const empty = slots.indexOf(null);
    const target = empty !== -1 ? empty : nextSlot;
    setSlots(slots.map((value, i) => (i === target ? model : value)));
    if (empty === -1) setNextSlot((nextSlot + 1) % MAX_SELECTED);
  };

  const angle = (index: number) =>
    (Math.PI * 2 * index) / skills.length - Math.PI / 2;
  const at = (index: number, value: number) => {
    const r = (value / 100) * RADIUS;
    return [CENTER + r * Math.cos(angle(index)), CENTER + r * Math.sin(angle(index))];
  };
  const ring = (value: number) =>
    skills.map((_, index) => at(index, value).join(",")).join(" ");

  const chosen = slots.flatMap((model, slot) => {
    const entry = entries.find((item) => item.model === model);
    return entry ? [{ entry, slot }] : [];
  });

  return (
    <div className={styles.wrap}>
      <fieldset className={styles.picker}>
        <legend className={styles.pickerLabel}>Pick two</legend>
        {entries.map((entry) => {
          const slot = slots.indexOf(entry.model);
          return (
            <label key={entry.model} className={styles.option} data-slot={slot}>
              <input
                type="checkbox"
                checked={slot !== -1}
                onChange={() => toggle(entry.model)}
              />
              <span className={styles.swatch} aria-hidden />
              <ModelName model={entry.model} provider={entry.provider} />
            </label>
          );
        })}
      </fieldset>

      <div className={styles.chart}>
        <ul className={styles.legend}>
          {chosen.map(({ entry, slot }) => (
            <li key={entry.model} data-slot={slot}>
              <span className={styles.swatch} aria-hidden />
              <ModelName model={entry.model} provider={entry.provider} />
            </li>
          ))}
        </ul>

        <svg
          viewBox={`0 0 ${SIZE} ${SIZE}`}
          className={styles.svg}
          role="img"
          aria-label={`Skill scores for ${chosen.map(({ entry }) => entry.model).join(" and ")}`}
        >
          {RINGS.map((value) => (
            <polygon key={value} points={ring(value)} className={styles.ring} />
          ))}
          {skills.map((skill, index) => {
            const [x, y] = at(index, 100);
            const [lx, ly] = at(index, 122);
            const cos = Math.cos(angle(index));
            return (
              <g key={skill.id}>
                <line x1={CENTER} y1={CENTER} x2={x} y2={y} className={styles.spoke} />
                <text
                  x={lx}
                  y={ly}
                  className={styles.axisLabel}
                  textAnchor={
                    Math.abs(cos) < 0.2 ? "middle" : cos > 0 ? "start" : "end"
                  }
                  dominantBaseline="middle"
                >
                  {skill.label}
                </text>
              </g>
            );
          })}

          {chosen.map(({ entry, slot }) => (
            <g key={entry.model} data-slot={slot} className={styles.series}>
              <polygon
                points={skills
                  .map((skill, index) =>
                    at(index, entry.skillScores[skill.id]).join(","),
                  )
                  .join(" ")}
                className={styles.shape}
              />
              {skills.map((skill, index) => {
                const [x, y] = at(index, entry.skillScores[skill.id]);
                return (
                  <circle key={skill.id} cx={x} cy={y} r={4.5} className={styles.dot}>
                    <title>
                      {`${entry.model}, ${skill.label}: ${entry.skillScores[skill.id]}`}
                    </title>
                  </circle>
                );
              })}
            </g>
          ))}
        </svg>

        <div className={styles.summary}>
          {chosen.map(({ entry }) => (
            <p key={entry.model}>
              <strong>
                <ModelName model={entry.model} provider={entry.provider} />
              </strong>
              : {describe(entry, skills)}
            </p>
          ))}
        </div>

        <div className={styles.tableScroll}>
          <table className={styles.table}>
            <thead>
              <tr>
                <th scope="col">Skill</th>
                {chosen.map(({ entry }) => (
                  <th key={entry.model} scope="col">
                    <ModelName model={entry.model} provider={entry.provider} />
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {skills.map((skill) => (
                <tr key={skill.id}>
                  <th scope="row">{skill.label}</th>
                  {chosen.map(({ entry }) => (
                    <td key={entry.model}>{entry.skillScores[skill.id]}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

export default RadarCompare;
