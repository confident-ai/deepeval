"use client";

import { useEffect, useRef, useState, type ReactNode } from "react";
import { useInView } from "@/src/components/Mascot/_useInView";
import Mascot, { type Character } from "@/src/components/Mascot";

import styles from "./MascotPeek.module.scss";

const FIRST_PEEK_DELAY_MS = 600;
const FIRST_PEEK_MS = 7000;
const PEEK_MS = 2200;
const PEEK_GAP_MS = 4800;
const PEEK_JITTER_MS = 3000;

interface MascotPeekProps {
  mascot: Character;
  children: ReactNode;
}

const MascotPeek = ({ mascot, children }: MascotPeekProps) => {
  const { speed: _speed, ...character } = mascot;
  const [hovered, setHovered] = useState(false);
  const [mascotHovered, setMascotHovered] = useState(false);
  const [timedPeek, setTimedPeek] = useState(false);
  const hasPeeked = useRef(false);
  const { ref, inView } = useInView({ threshold: 0.5 });

  useEffect(() => {
    if (!inView) return;
    let hide: ReturnType<typeof setTimeout>;
    let next: ReturnType<typeof setTimeout>;
    const peek = () => {
      const duration = hasPeeked.current ? PEEK_MS : FIRST_PEEK_MS;
      hasPeeked.current = true;
      setTimedPeek(true);
      hide = setTimeout(() => setTimedPeek(false), duration);
      next = setTimeout(
        peek,
        duration + PEEK_GAP_MS + Math.random() * PEEK_JITTER_MS
      );
    };
    next = setTimeout(peek, FIRST_PEEK_DELAY_MS);
    return () => {
      clearTimeout(hide);
      clearTimeout(next);
      setTimedPeek(false);
    };
  }, [inView]);

  const hideoutClass = [
    styles.hideout,
    (hovered || timedPeek) && styles.out,
    mascotHovered && styles.waving,
  ]
    .filter(Boolean)
    .join(" ");

  return (
    <span
      ref={ref}
      className={styles.peek}
      onPointerEnter={() => setHovered(true)}
      onPointerLeave={() => setHovered(false)}
      onFocus={() => setHovered(true)}
      onBlur={() => setHovered(false)}
    >
      {children}
      <span className={hideoutClass} aria-hidden>
        <span
          className={styles.mascot}
          onPointerEnter={() => setMascotHovered(true)}
          onPointerLeave={() => setMascotHovered(false)}
        >
          <Mascot {...character} animated look="left" hover="wave" />
        </span>
      </span>
    </span>
  );
};

export default MascotPeek;
