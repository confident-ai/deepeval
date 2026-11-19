import Link from "next/link";
import AnnouncementBar from "./AnnouncementBar";
import styles from "./MeetupBanner.module.scss";

// Remove after the event: delete this folder, put the JevEval fumadocs
// <Banner id="docs-announcement"> back in lib/section.tsx, and remove
// <MeetupBanner> from src/layouts/HomePageShell and bannerPrepaintScript
// from app/layout.tsx.
//
// Pages are prerendered, so the server cannot know the date. Both bars are
// rendered, and bannerPrepaintScript (inline in <head>) adds
// MEETUP_ENDED_CLASS to <html> before first paint once the event starts.
// The module CSS then shows the right bar, with no flash or layout shift.
const MEETUP_BANNER_ID = "sf-meetup-2026-11-20";
const JEVEVAL_BANNER_ID = "docs-announcement";
const MEETUP_ENDED_CLASS = "sf-meetup-ended";
const EVENT_START_MS = Date.parse("2026-11-20T17:00:00-08:00");
const RSVP_URL =
  "https://luma.com/tzyy1h0e?utm_source=deepeval.com&utm_medium=banner&utm_campaign=sf_meetup";

const BANNER_IDS = JSON.stringify([MEETUP_BANNER_ID, JEVEVAL_BANNER_ID]);
export const bannerPrepaintScript = `(function(){var h=document.documentElement;if(Date.now()>=${EVENT_START_MS})h.classList.add("${MEETUP_ENDED_CLASS}");try{${BANNER_IDS}.forEach(function(id){if(localStorage.getItem("nd-banner-"+id)==="true")h.classList.add("nd-banner-"+id)})}catch(e){}})()`;

type MeetupBannerProps = {
  // "section": the sticky docs-layout slot; the JevEval bar takes over once
  // the event starts.
  // "shell": HomePageShell pages (/ and /enterprise); the bar scrolls away so
  // it never covers the sticky top nav, and nothing replaces it.
  placement: "section" | "shell";
};

const MeetupBanner: React.FC<MeetupBannerProps> = ({ placement }) => {
  const isSection = placement === "section";
  return (
    <>
      <AnnouncementBar
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
      </AnnouncementBar>
      {isSection ? (
        <AnnouncementBar id={JEVEVAL_BANNER_ID} height="30px" changeLayout>
          💥 Introducing JevEval: Jev-as-a-Judge for LLM evaluation.{" "}
          <Link href="/blog/introducing-jev-as-a-judge">Read the post →</Link>
        </AnnouncementBar>
      ) : null}
    </>
  );
};

export default MeetupBanner;
