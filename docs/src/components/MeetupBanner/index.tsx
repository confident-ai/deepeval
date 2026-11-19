"use client";

import { useEffect, useState, type ReactNode } from "react";
import { Banner } from "fumadocs-ui/components/banner";
import styles from "./MeetupBanner.module.scss";

// SF meetup announcement. Until the event starts it takes the banner slot;
// afterwards the slot shows `fallback` again (the regular docs announcement).
// Delete this component and restore the plain fallback after the event.
const EVENT_ID = "sf-meetup-2026-11-20";
const EVENT_START = new Date("2026-11-20T17:00:00-08:00");
const RSVP_URL =
  "https://luma.com/tzyy1h0e?utm_source=deepeval.com&utm_medium=banner&utm_campaign=sf_meetup";

type MeetupBannerProps = {
  // "section": the sticky docs-layout slot. "shell": the homepage, which
  // scrolls the bar away so it never sits over the sticky home nav.
  placement: "section" | "shell";
  fallback?: ReactNode;
};

const MeetupBanner: React.FC<MeetupBannerProps> = ({ placement, fallback = null }) => {
  // Checked in the browser: pages are pre-built, so a server-side date
  // check would freeze at build time.
  const [eventStarted, setEventStarted] = useState(false);
  useEffect(() => {
    setEventStarted(Date.now() >= EVENT_START.getTime());
  }, []);

  if (eventStarted) return <>{fallback}</>;

  const isSection = placement === "section";
  return (
    <Banner
      id={EVENT_ID}
      height={isSection ? "30px" : "2.5rem"}
      changeLayout={isSection}
      className={isSection ? styles.banner : `${styles.banner} ${styles.static}`}
    >
      <span>
        <span className={styles.wideOnly}>DeepEval </span>SF meetup, this Friday 5pm.{" "}
        <a href={RSVP_URL} target="_blank" rel="noopener noreferrer" className={styles.link}>
          RSVP on Luma →
        </a>
      </span>
    </Banner>
  );
};

export default MeetupBanner;
