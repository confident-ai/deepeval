import { ArrowUpRight } from "lucide-react";
import { PrimaryButton, SecondaryButton } from "@site/src/components/Buttons";
import { PauseOffscreen } from "@site/src/components/PauseOffscreen";
import MascotRail, { type Walker } from "@site/src/components/MascotRail";
import styles from "@site/src/sections/home/HomeSection.module.scss";
import { communityEvents, countCities } from "@/lib/community/events";
import { contributorCount, contributionCount } from "@/lib/community/contributors";
import { communitySubmitUrl } from "@/lib/shared";

// A meetup is a crowd, so the rail walks five mascots in different colors
// and hats instead of the homepage's single one.
const WALKERS: Walker[] = [
  { color: "color-mix(in srgb, #760eff 45%, white)", speed: 5, hover: "wave", at: 0.08 },
  { color: "#ff8a3d", speed: 4, accessory: "headphones", hover: "wave", at: 0.3 },
  { color: "#2ec4a6", speed: 6, accessory: "snapback", hover: "wave", at: 0.5 },
  { color: "#ff5c8a", speed: 3, accessory: "sprout", hover: "wave", at: 0.7 },
  { color: "#f2c230", speed: 5, accessory: "gradcap", hover: "wave", at: 0.9 },
];

const compact = (n: number) =>
  new Intl.NumberFormat("en-US", { notation: "compact", maximumFractionDigits: 1 }).format(n);

const CommunityHeroSection: React.FC = () => {
  const cities = countCities(communityEvents);
  const bannerItems = [
    `${contributorCount} contributors`,
    `${compact(contributionCount)} contributions on GitHub`,
    "Used by 150K+ developers",
    `Meetups in ${cities} ${cities === 1 ? "city" : "cities"}, yours next`,
  ];

  return (
    <section className={styles.hero}>
      <div className={styles.main}>
        <h1 className={styles.title}>Host a DeepEval meetup in your city.</h1>

        <p className={styles.description}>
          Engineers around the world run meetups about DeepEval and LLM evals.
          List your event here, and we help with swag, a remote speaker and
          free credits.
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
          <SecondaryButton href="#list-your-event-in-four-steps">
            How to host
          </SecondaryButton>
        </div>
      </div>

      <div className={styles.bannerWalkway}>
        <MascotRail walkers={WALKERS} edge="top" />
        <PauseOffscreen className={styles.banner} aria-label="The DeepEval community in numbers">
          <div className={styles.bannerTrack}>
            {[...bannerItems, ...bannerItems].map((item, i) => (
              <span key={i} className={styles.bannerItem} aria-hidden={i >= bannerItems.length}>
                {item}
              </span>
            ))}
          </div>
        </PauseOffscreen>
      </div>
    </section>
  );
};

export default CommunityHeroSection;
