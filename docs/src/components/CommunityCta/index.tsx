import Link from "next/link";
import { discordUrl, redditUrl } from "@/lib/shared";
import { DiscordMark } from "@/src/components/DiscordButton";
import { RedditMark } from "@/src/components/RedditButton";
import styles from "./CommunityCta.module.scss";

type CommunityCtaProps = {
  label?: string;
  layout?: "full" | "inline";
};

const CommunityCta: React.FC<CommunityCtaProps> = ({
  label = "Join",
  layout = "full",
}) => {
  return (
    <div className={styles.root} data-layout={layout}>
      <span className={styles.label}>{label}</span>
      <div className={styles.actions}>
        <Link
          href={discordUrl}
          target="_blank"
          rel="noopener noreferrer"
          className={`${styles.brand} ${styles.discord}`}
          aria-label="Join our Discord community"
          data-callout
          data-button
        >
          <DiscordMark />
          <span>Discord</span>
        </Link>
        <Link
          href={redditUrl}
          target="_blank"
          rel="noopener noreferrer"
          className={`${styles.brand} ${styles.reddit}`}
          aria-label="Join our Subreddit"
          data-callout
          data-button
        >
          <RedditMark />
          <span>r/deepeval</span>
        </Link>
      </div>
    </div>
  );
};

export default CommunityCta;
