import { ArrowUpRight } from "lucide-react";
import {
  formatEventDate,
  splitEvents,
  type CommunityEvent,
} from "@/lib/community/events";
import { communitySubmitUrl } from "@/lib/shared";
import styles from "./CommunityEvents.module.scss";

type CommunityEventsProps = {
  when: "upcoming" | "past";
};

function EventRow({ event, past }: { event: CommunityEvent; past: boolean }) {
  const { day, time } = formatEventDate(event);
  const place =
    event.city.toLowerCase() === "online"
      ? "Online"
      : `${event.city}, ${event.country}`;

  return (
    <li className={styles.row}>
      <a
        className={styles.link}
        href={event.url}
        target="_blank"
        rel="noopener noreferrer"
        data-past={past || undefined}
        // data-card opts out of the global .prose link style (app/global.css).
        data-card
      >
        <div className={styles.when}>
          <span className={styles.day}>{day}</span>
          <span className={styles.time}>{time}</span>
        </div>
        <div className={styles.what}>
          <span className={styles.title}>
            {event.title}
            {event.official && <span className={styles.tag}>Official</span>}
          </span>
          <span className={styles.talk}>{event.talk}</span>
          <span className={styles.meta}>
            {place} · Hosted by {event.host}
          </span>
        </div>
        <ArrowUpRight className={styles.arrow} aria-hidden />
      </a>
    </li>
  );
}

const CommunityEvents: React.FC<CommunityEventsProps> = ({ when }) => {
  const { upcoming, past } = splitEvents();
  const list = when === "upcoming" ? upcoming : past;

  if (list.length === 0) {
    return (
      <div className={`not-prose ${styles.empty}`}>
        {when === "upcoming" ? (
          <>
            No upcoming meetups yet.{" "}
            <a href={communitySubmitUrl} target="_blank" rel="noopener noreferrer">
              Host the first one in your city
            </a>
            .
          </>
        ) : (
          "Past meetups appear here."
        )}
      </div>
    );
  }

  return (
    <ul className={`not-prose ${styles.list}`}>
      {list.map((event) => (
        <EventRow key={event.id} event={event} past={when === "past"} />
      ))}
    </ul>
  );
};

export default CommunityEvents;
