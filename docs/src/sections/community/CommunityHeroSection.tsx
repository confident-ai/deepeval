import { ArrowUpRight } from "lucide-react";
import { PrimaryButton, SecondaryButton } from "@site/src/components/Buttons";
import { DiscordMark } from "@/src/components/DiscordButton";
import styles from "@site/src/sections/home/HomeSection.module.scss";
import { communityEvents, countCities } from "@/lib/community/events";
import { communitySubmitUrl, discordUrl } from "@/lib/shared";
import communityStyles from "./CommunityHeroSection.module.scss";

const CommunityHeroSection: React.FC = () => {
  const cities = countCities(communityEvents);

  return (
    <section className={styles.hero}>
      <div className={styles.main}>
        <h1 className={styles.title}>Host a DeepEval meetup in your city.</h1>

        <p className={styles.description}>
          Engineers around the world run meetups about DeepEval and LLM evals.
          List your event here, and we help with swag, a remote speaker and
          Confident AI credits.
        </p>

        <div className={styles.actions}>
          <PrimaryButton
            href={communitySubmitUrl}
            target="_blank"
            rel="noopener noreferrer"
            endIcon={<ArrowUpRight aria-hidden />}
          >
            Submit your event
          </PrimaryButton>
          <SecondaryButton
            href={discordUrl}
            target="_blank"
            rel="noopener noreferrer"
            startIcon={<DiscordMark />}
          >
            Join Discord
          </SecondaryButton>
        </div>
      </div>

      <dl className={communityStyles.stats}>
        <div className={communityStyles.stat}>
          <dt>Events</dt>
          <dd>{communityEvents.length}</dd>
        </div>
        <div className={communityStyles.stat}>
          <dt>Cities</dt>
          <dd>{cities}</dd>
        </div>
        <div className={communityStyles.stat}>
          <dt>Listed in</dt>
          <dd>2 days</dd>
        </div>
      </dl>
    </section>
  );
};

export default CommunityHeroSection;
