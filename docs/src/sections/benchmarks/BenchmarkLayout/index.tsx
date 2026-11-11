"use client";

import { useEffect, useState, type ReactNode } from "react";
import styles from "./BenchmarkLayout.module.scss";

export type SectionLink = {
  id: string;
  label: string;
};

type BenchmarkLayoutProps = {
  sections: SectionLink[];
  children: ReactNode;
};

const BenchmarkLayout: React.FC<BenchmarkLayoutProps> = ({
  sections,
  children,
}) => {
  const [activeId, setActiveId] = useState(sections[0]?.id);

  useEffect(() => {
    const targets = sections
      .map((section) => document.getElementById(section.id))
      .filter((element): element is HTMLElement => element !== null);

    // The active section is the last one whose top has passed the upper
    // third of the viewport.
    const observer = new IntersectionObserver(
      () => {
        const line = window.innerHeight / 3;
        const passed = targets.filter(
          (element) => element.getBoundingClientRect().top <= line,
        );
        setActiveId((passed[passed.length - 1] ?? targets[0])?.id);
      },
      { rootMargin: "0px 0px -66% 0px", threshold: [0, 1] },
    );

    targets.forEach((element) => observer.observe(element));
    return () => observer.disconnect();
  }, [sections]);

  return (
    <div className={styles.layout}>
      <nav className={styles.nav} aria-label="On this page">
        <p className={styles.navTitle}>On this page</p>
        <ul className={styles.navList}>
          {sections.map((section) => (
            <li key={section.id}>
              <a
                href={`#${section.id}`}
                className={styles.navLink}
                aria-current={section.id === activeId ? "location" : undefined}
              >
                {section.label}
              </a>
            </li>
          ))}
        </ul>
      </nav>
      <div className={styles.content}>{children}</div>
    </div>
  );
};

export default BenchmarkLayout;
