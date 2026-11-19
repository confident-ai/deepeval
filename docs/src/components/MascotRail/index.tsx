"use client";

import { useEffect, useRef, useState } from "react";
import { useInView } from "@/src/components/Mascot/_useInView";
import Mascot, { type Character } from "@/src/components/Mascot";
import { CLICKS } from "@/src/components/Mascot/_cast";
import styles from "./MascotRail.module.scss";

export type WalkerIdle = "walk" | "stand" | "float";

export type Walker = Character & {
  idle?: WalkerIdle;
  at?: number;
};

interface MascotRailProps {
  walkers: readonly Walker[];
  edge?: "top" | "bottom";
}

type Direction = 1 | -1;

interface WalkerState {
  x: number;
  still: boolean;
  dir: Direction;
  resting: boolean;
  restUntil: number;
  turnAfterRest: boolean;
}

const DEFAULT_SPEED = 5;
const STRIDE = 4;
const REST_CHANCE = 0.06;

const restFor = () => 1500 + Math.random() * 2000;

const walks = (walker: Walker) => (walker.idle ?? "walk") === "walk";

const MascotRail = ({ walkers, edge = "bottom" }: MascotRailProps) => {
  const { ref: inViewRef, inView } = useInView({ rootMargin: "100px" });
  const railRef = useRef<HTMLDivElement | null>(null);
  const nodes = useRef<(HTMLDivElement | null)[]>([]);
  const held = useRef<boolean[]>([]);
  const state = useRef<WalkerState[]>([]);
  const [view, setView] = useState(() =>
    walkers.map((_, i) => ({
      walking: false,
      dir: (i % 2 === 0 ? 1 : -1) as Direction,
    }))
  );

  useEffect(() => {
    const rail = railRef.current;
    const host = rail?.offsetParent;
    if (edge !== "top" || !rail || !host) return;
    const border = parseFloat(getComputedStyle(host).borderTopWidth) || 0;
    rail.style.bottom = `calc(100% + ${border}px)`;
  }, [edge]);

  useEffect(() => {
    const rail = railRef.current;
    if (!inView || !rail) return;
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
    if (!walkers.some(walks)) return;

    const size = () => nodes.current[0]?.offsetWidth || 40;

    if (state.current.length !== walkers.length) {
      const railLeft = rail.getBoundingClientRect().left;
      state.current = walkers.map((_, i) => ({
        x:
          (nodes.current[i]?.getBoundingClientRect().left ?? railLeft) -
          railLeft,
        still: !walks(walkers[i]),
        dir: i % 2 === 0 ? 1 : -1,
        resting: false,
        restUntil: 0,
        turnAfterRest: false,
      }));
      nodes.current.forEach((node) => node && (node.style.left = "0"));
    }
    let shown = view;

    const rest = (w: WalkerState, now: number, turn: boolean) => {
      w.resting = true;
      w.restUntil = now + restFor();
      w.turnAfterRest = turn;
    };

    let frame = 0;
    let last = performance.now();

    const tick = (now: number) => {
      const dt = Math.min(now - last, 50) / 1000;
      last = now;
      const width = size();
      const pixel = width / 16;
      const limit = Math.max(0, rail.clientWidth - width);
      const walkersState = state.current;

      walkersState.forEach((w, i) => {
        if (w.still) {
          const at = walkers[i].at;
          w.x = at === undefined ? Math.min(w.x, limit) : at * limit;
          return;
        }
        if (held.current[i]) return;
        if (w.resting) {
          if (now < w.restUntil) return;
          w.resting = false;
          if (w.turnAfterRest) w.dir = w.dir === 1 ? -1 : 1;
        }
        w.x += w.dir * (walkers[i].speed ?? DEFAULT_SPEED) * pixel * dt;
        if (w.x <= 0 || w.x >= limit) {
          w.x = Math.min(Math.max(w.x, 0), limit);
          rest(w, now, true);
        } else if (Math.random() < REST_CHANCE * dt) {
          rest(w, now, Math.random() < 0.5);
        }
      });

      walkersState.forEach((a, i) =>
        walkersState.forEach((b, j) => {
          if (i === j || a.still || a.resting || held.current[i]) return;
          const gap = b.x - a.x;
          if (Math.abs(gap) < width * 0.9 && Math.sign(gap) === a.dir) {
            a.dir = a.dir === 1 ? -1 : 1;
          }
        })
      );

      walkersState.forEach((w, i) => {
        const node = nodes.current[i];
        if (node) {
          const x = Math.round(w.x / pixel) * pixel;
          node.style.transform = `translateX(${x}px)`;
        }
      });

      const next = walkersState.map((w, i) => ({
        walking: !w.still && !w.resting && !held.current[i],
        dir: w.dir,
      }));
      if (
        next.some(
          (v, i) => v.walking !== shown[i].walking || v.dir !== shown[i].dir
        )
      ) {
        shown = next;
        setView(next);
      }

      frame = requestAnimationFrame(tick);
    };

    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [inView, walkers]);

  return (
    <div
      ref={(node) => {
        railRef.current = node;
        inViewRef(node);
      }}
      className={`${styles.rail} ${edge === "top" ? styles.top : ""}`}
    >
      {walkers.map(({ speed: _speed, idle, at, ...mascot }, i) => {
        const spot = at ?? (i + 1) / (walkers.length + 1);
        return (
          <div
            key={i}
            ref={(node) => {
              nodes.current[i] = node;
            }}
            className={styles.walker}
            style={{
              ["--mascot-step" as string]: `${
                (2 * STRIDE) / (walkers[i].speed ?? DEFAULT_SPEED)
              }s`,
              left: `${spot * 100}%`,
              transform: `translateX(-${spot * 100}%)`,
            }}
            onPointerEnter={() => (held.current[i] = true)}
            onPointerLeave={() => (held.current[i] = false)}
          >
            <Mascot
              animated
              animationDelay={i * -0.53}
              pose={view[i].walking ? "walk" : "idle"}
              motion={idle === "float" ? "fly" : undefined}
              look={
                view[i].walking
                  ? view[i].dir === 1
                    ? "right"
                    : "left"
                  : "center"
              }
              hover="wave"
              click={CLICKS}
              {...mascot}
            />
          </div>
        );
      })}
    </div>
  );
};

export default MascotRail;
