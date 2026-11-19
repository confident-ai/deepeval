import type { ReactNode } from "react";
import { Banner } from "fumadocs-ui/components/banner";
import BannerDismissSync from "./BannerDismissSync";
import styles from "./MeetupBanner.module.scss";

// Remove after the event: delete this folder, put the JevEval <Banner> back
// in lib/section.tsx, and remove <MeetupBanner> from
// src/layouts/HomePageShell and meetupEndedScript from app/layout.tsx.
//
// Pages are prerendered, so the server cannot know the date. Both bars are
// rendered, and meetupEndedScript (inline in <head>) adds MEETUP_ENDED_CLASS
// to <html> before first paint once the event starts. The module CSS then
// shows the right bar, with no flash or layout shift.
const MEETUP_BANNER_ID = "sf-meetup-2026-11-20";
const MEETUP_ENDED_CLASS = "sf-meetup-ended";
const EVENT_START_MS = Date.parse("2026-11-20T17:00:00-08:00");
const RSVP_URL =
  "https://luma.com/tzyy1h0e?utm_source=deepeval.com&utm_medium=banner&utm_campaign=sf_meetup";

export const meetupEndedScript = `if(Date.now()>=${EVENT_START_MS})document.documentElement.classList.add("${MEETUP_ENDED_CLASS}")`;

type MeetupBannerProps = {
  // "section": the sticky docs-layout slot; `fallback` (the JevEval banner,
  // id "docs-announcement") takes over once the event starts.
  // "shell": HomePageShell pages (/ and /enterprise); the bar scrolls away so
  // it never covers the sticky top nav, and nothing replaces it.
  placement: "section" | "shell";
  fallback?: ReactNode;
};

const MeetupBanner: React.FC<MeetupBannerProps> = ({ placement, fallback = null }) => {
  const isSection = placement === "section";
  return (
    <>
      <BannerDismissSync />
      <Banner
        id={MEETUP_BANNER_ID}
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
      {isSection ? fallback : null}
    </>
  );
};

export default MeetupBanner;
