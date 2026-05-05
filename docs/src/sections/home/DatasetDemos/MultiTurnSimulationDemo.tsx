"use client";

import { useEffect, useMemo, useState } from "react";
import styles from "./DatasetDemos.module.scss";

type Turn = {
  role: "user" | "agent";
  text: string;
};

const TURNS: Turn[] = [
  {
    role: "user",
    text: "I want to return something I bought last week.",
  },
  {
    role: "agent",
    text: "I can help. Could you share your order number?",
  },
  {
    role: "user",
    text: "It's #9281 — but I misplaced the packaging. Does that matter?",
  },
  {
    role: "agent",
    text: "No worries. Original packaging isn't required. I'll initiate the return for #9281 right now.",
  },
];

const METRICS = [
  { name: "Relevancy", score: "0.93" },
  { name: "Helpfulness", score: "0.91" },
  { name: "Policy adherence", score: "1.00" },
];

const STAGES = [
  "Pondering scenario",
  "Analyzing user profile",
  "Simulating user response",
] as const;

const UNDERSTANDING_MS = 1000;
const PROFILE_MS = 1000;
const CHAR_MS = 10;
const TURN_GAP_MS = 220;
const SCORE_DELAY_MS = 300;

/* --------------------------- TurnView --------------------------- */

type TurnViewProps = {
  turn: Turn;
  revealed: number;
  reducedMotion: boolean;
};

const TurnView: React.FC<TurnViewProps> = ({ turn, revealed, reducedMotion }) => {
  const done = revealed >= turn.text.length;

  return (
    <div
      className={`${styles.turn} ${
        styles[`turn_${turn.role}` as keyof typeof styles]
      }`}
    >
      <span className={styles.turnLabel}>
        {turn.role === "user" ? "USER · simulated" : "AGENT"}
      </span>
      <div className={styles.bubble}>
        <span>{turn.text.slice(0, revealed)}</span>
        {!done && <span className={styles.caret} aria-hidden />}
      </div>
    </div>
  );
};

/* ----------------------- MultiTurnSimulationDemo ----------------------- */

export const MultiTurnSimulationDemo: React.FC = () => {
  const reducedMotion = useMemo(() => {
    if (typeof window === "undefined") return false;
    return window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  }, []);

  const [activeStage, setActiveStage] = useState(
    reducedMotion ? STAGES.length : 0
  );
  const [visibleTurns, setVisibleTurns] = useState(
    reducedMotion ? TURNS.length : 0
  );
  const [currentTurnIndex, setCurrentTurnIndex] = useState(0);
  const [revealedCounts, setRevealedCounts] = useState(
    reducedMotion ? TURNS.map((turn) => turn.text.length) : TURNS.map(() => 0)
  );
  const [showScore, setShowScore] = useState(reducedMotion);
  const isDone = activeStage >= STAGES.length;

  useEffect(() => {
    if (reducedMotion) return;

    if (activeStage === 0) {
      const id = setTimeout(() => setActiveStage(1), UNDERSTANDING_MS);
      return () => clearTimeout(id);
    }

    if (activeStage === 1) {
      const id = setTimeout(() => {
        setActiveStage(2);
        setVisibleTurns(1);
      }, PROFILE_MS);
      return () => clearTimeout(id);
    }

    if (activeStage !== 2) return;

    if (currentTurnIndex >= TURNS.length) {
      if (showScore) return;
      const id = setTimeout(() => {
        setShowScore(true);
        setActiveStage(STAGES.length);
      }, SCORE_DELAY_MS);
      return () => clearTimeout(id);
    }

    const currentTurn = TURNS[currentTurnIndex];
    const revealed = revealedCounts[currentTurnIndex];

    if (revealed < currentTurn.text.length) {
      const id = setTimeout(() => {
        setRevealedCounts((counts) =>
          counts.map((count, i) =>
            i === currentTurnIndex
              ? Math.min(count + 1, currentTurn.text.length)
              : count
          )
        );
      }, CHAR_MS);
      return () => clearTimeout(id);
    }

    const id = setTimeout(() => {
      const nextIndex = currentTurnIndex + 1;
      setCurrentTurnIndex(nextIndex);
      if (nextIndex < TURNS.length) {
        setVisibleTurns(nextIndex + 1);
      }
    }, TURN_GAP_MS);
    return () => clearTimeout(id);
  }, [activeStage, currentTurnIndex, revealedCounts, reducedMotion, showScore]);

  return (
    <div className={styles.panel} data-done={isDone ? "true" : undefined}>
      <ol className={styles.stages} aria-label="simulation pipeline">
        {STAGES.map((stage, i) => {
          const state =
            i < activeStage ? "done" : i === activeStage ? "active" : "pending";
          return (
            <li
              key={stage}
              className={`${styles.stage} ${
                styles[`stage_${state}` as keyof typeof styles]
              }`}
              aria-current={state === "active" ? "step" : undefined}
            >
              <span className={styles.stageMark} aria-hidden>
                {state === "done" ? "✓" : state === "active" ? "●" : "○"}
              </span>
              <span className={styles.stageLabel}>{stage}</span>
            </li>
          );
        })}
      </ol>

      <div className={styles.conversation}>
        {TURNS.slice(0, visibleTurns).map((turn, i) => (
          <TurnView
            key={i}
            turn={turn}
            revealed={revealedCounts[i]}
            reducedMotion={reducedMotion}
          />
        ))}

        {showScore && (
          <div className={styles.scoreRow}>
            {METRICS.map((m) => (
              <div key={m.name} className={styles.scoreItem}>
                <span className={styles.scoreName}>{m.name}</span>
                <span className={styles.scoreValue}>{m.score}</span>
              </div>
            ))}
            <div className={styles.scoreAllPassed}>all passed</div>
          </div>
        )}
      </div>
    </div>
  );
};
