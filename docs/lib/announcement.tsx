import Link from "next/link";
import { AnnouncementBar } from "./announcement-bar";

// SF meetup banner. Pages are prerendered, so meetupEndScript (in <head>)
// marks <html> once the event is over and global.css swaps back to JevEval.
// After the event, remove this file, its imports, and the meetup block in
// app/global.css (it hides #docs-announcement until meetup-ended is set).
// Keep lib/banner-restore.ts and lib/announcement-bar.tsx: the JevEval bar
// uses them.
export const meetupEndScript = `if(Date.now()>=${Date.parse("2026-11-20T19:00:00-08:00")})document.documentElement.classList.add("meetup-ended")`;

export function MeetupBanner({ changeLayout = true }: { changeLayout?: boolean }) {
  return (
    <AnnouncementBar id="sf-meetup" height="30px" changeLayout={changeLayout}>
      <span>
        📍 Join the 1st DeepEval meetup in SF, this Friday 5pm PT.{" "}
        <Link href="https://luma.com/tzyy1h0e?utm_source=deepeval.com&utm_medium=banner&utm_campaign=sf_meetup">
          RSVP on Luma →
        </Link>
      </span>
    </AnnouncementBar>
  );
}
