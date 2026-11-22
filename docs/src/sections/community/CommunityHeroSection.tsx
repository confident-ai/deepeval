import { ArrowUpRight, Gift, GitCommitHorizontal, Mic, Ticket, Users, Code } from "lucide-react";
import { PrimaryButton, SecondaryButton } from "@site/src/components/Buttons";
import { PauseOffscreen } from "@site/src/components/PauseOffscreen";
import MascotRail, { type Walker } from "@site/src/components/MascotRail";
import type { MascotHolding } from "@site/src/components/Mascot";
import contributorCountries from "@/lib/generated/contributor-countries.json";
import ContributorMarquee from "./ContributorMarquee";
import styles from "@site/src/sections/home/HomeSection.module.scss";
import local from "./CommunityHero.module.scss";
import MeetupGlobe from "./MeetupGlobe";
import { countryFlag, countryName, flagOf, splitEvents, type CommunityEvent } from "@/lib/community/events";
import { contributorCount, contributionCount } from "@/lib/community/contributors";
import { communitySubmitUrl } from "@/lib/shared";

const compact = (n: number) =>
  new Intl.NumberFormat("en-US", { notation: "compact", maximumFractionDigits: 1 }).format(n);

// The walkers stroll the strip like on the homepage, in their usual colors.
// The host carries the microphone; the others are contributors from the
// countries with the most contributors, each holding something their
// culture loves. Hover one for who it is.
const CULTURE: Record<string, MascotHolding> = {
  US: "hotdog",
  IN: "cricket",
  CN: "lantern",
  DE: "pretzel",
};
const topCountries = (contributorCountries as { code: string; logins: string[] }[]).filter(
  (c) => CULTURE[c.code] && c.logins.length > 0
);
const carrying = (i: number) => {
  const c = topCountries[i];
  return c
    ? {
        holding: CULTURE[c.code],
        label: `@${c.logins[0]} · ${flagOf(c.code)} ${countryName(c.code)}`,
      }
    : {};
};

const WALKERS: Walker[] = [
  { color: "color-mix(in srgb, #760eff 45%, white)", holding: "mic", label: "The host, with the mic", speed: 5, hover: "wave", at: 0.08 },
  { color: "#ff8a3d", accessory: "headphones", ...carrying(0), speed: 4, hover: "wave", at: 0.3 },
  { color: "#2ec4a6", accessory: "snapback", ...carrying(1), speed: 6, hover: "wave", at: 0.5 },
  { color: "#ff5c8a", accessory: "sprout", ...carrying(2), speed: 3, hover: "wave", at: 0.7 },
  { color: "#f2c230", accessory: "gradcap", ...carrying(3), speed: 5, hover: "wave", at: 0.9 },
];

type StripItem = { key: string; icon?: React.ReactNode; flag?: string; text: string };

const shortDate = (e: CommunityEvent) =>
  new Intl.DateTimeFormat("en-US", { month: "short", day: "numeric", timeZone: e.timezone }).format(
    new Date(e.start)
  );

const meetupItem = (e: CommunityEvent, next: boolean): StripItem => ({
  key: e.id,
  flag: countryFlag(e.country) || undefined,
  text: `${next ? "Next: " : ""}${e.city} · ${shortDate(e)}`,
});


const CommunityHeroSection: React.FC = () => {
  const { upcoming, past } = splitEvents();
  const icon = { size: 13, strokeWidth: 1.75, "aria-hidden": true } as const;

  // Three kinds of items, interleaved: real meetups (with flags), numbers
  // about the project, and what a host gets from us.
  const meetups = [...upcoming.map((e) => meetupItem(e, true)), ...past.slice(0, 3).map((e) => meetupItem(e, false))];
  const stats: StripItem[] = [
    { key: "contributors", icon: <Users {...icon} />, text: `${contributorCount} contributors` },
    { key: "contributions", icon: <GitCommitHorizontal {...icon} />, text: `${compact(contributionCount)} contributions` },
    { key: "developers", icon: <Code {...icon} />, text: "150K+ developers" },
  ];
  const perks: StripItem[] = [
    { key: "swag", icon: <Gift {...icon} />, text: "Free swag" },
    { key: "speaker", icon: <Mic {...icon} />, text: "Remote speaker" },
    { key: "credits", icon: <Ticket {...icon} />, text: "Free credits" },
  ];
  const stripItems: StripItem[] = [];
  for (let i = 0; i < Math.max(meetups.length, stats.length, perks.length); i++) {
    for (const list of [meetups, stats, perks]) if (list[i]) stripItems.push(list[i]);
  }

  return (
    <section className={`${styles.hero} ${local.hero}`}>
      <div className={`${styles.main} ${local.main}`}>
        <h1 className={styles.title}>Host a DeepEval{"\u00a0"}meetup in your{"\u00a0"}city.</h1>

        <p className={styles.description}>
          Engineers get together to show how they test and debug AI agents.
          Host one in your city, and we'll help with swag, speakers and
          credits.
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
            Full guide
          </SecondaryButton>
        </div>
      </div>

      <div className={styles.bannerWalkway}>
        <MascotRail walkers={WALKERS} edge="top" />
        <PauseOffscreen className={styles.banner} aria-label="The DeepEval community in numbers">
          <div className={styles.bannerTrack}>
            {[...stripItems, ...stripItems].map((item, i) => (
              <span key={`${item.key}-${i}`} className={styles.bannerItem} aria-hidden={i >= stripItems.length}>
                <span className={local.stripItem}>
                  {item.flag ? <span className={local.stripFlag}>{item.flag}</span> : item.icon}
                  {item.text}
                </span>
              </span>
            ))}
          </div>
        </PauseOffscreen>
      </div>

      <MeetupGlobe />
      <ContributorMarquee />
    </section>
  );
};

export default CommunityHeroSection;
