import { ArrowUpRight } from "lucide-react";
import {
  countryFlag,
  formatEventDate,
  splitEvents,
  type CommunityEvent,
} from "@/lib/community/events";
import styles from "./CommunityEvents.module.scss";

type CommunityEventsProps = {
  when: "upcoming" | "past";
};

const isOnline = (e: CommunityEvent) => e.city.toLowerCase() === "online";

function EventRow({ event, past }: { event: CommunityEvent; past: boolean }) {
  const { day, time } = formatEventDate(event);
  const place =
    isOnline(event)
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
        <div className={styles.cover}>
          {event.image ? (
            // eslint-disable-next-line @next/next/no-img-element
            <img src={event.image} alt={`${event.title} cover`} loading="lazy" />
          ) : (
            <span className={styles.coverFallback} aria-hidden>
              {event.city.slice(0, 2).toUpperCase()}
            </span>
          )}
        </div>
        <div className={styles.what}>
          <span className={styles.when}>
            <span className={styles.day}>{day}</span>
            <span className={styles.time}>{time}</span>
          </span>
          <span className={styles.title}>
            {event.title}
            {event.official && <span className={styles.tag}>Official</span>}
          </span>
          <span className={styles.talk}>{event.talk}</span>
          <span className={styles.meta}>
            {isOnline(event) ? null : (
              <span className={styles.flag} role="img" aria-label={event.country}>
                {countryFlag(event.country)}
              </span>
            )}
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
            {/* Same target as the hero's "Full guide" button. */}
            <a href="#list-your-event-in-four-steps">Host the first one in your city</a>.
          </>
        ) : (
          "Past meetups will show up here."
        )}
      </div>
    );
  }

  return (
    <div className={`not-prose ${styles.window}`}>
      <div className={styles.chrome} aria-hidden>
        <span className={styles.dots}>
          <i />
          <i />
          <i />
        </span>
        <span className={styles.file}>meetups/{when}.json</span>
        <span className={styles.count}>
          {list.length} {list.length === 1 ? "event" : "events"}
        </span>
      </div>
      <ul className={styles.list}>
        {list.map((event) => (
          <EventRow key={event.id} event={event} past={when === "past"} />
        ))}
      </ul>
    </div>
  );
};

export default CommunityEvents;
