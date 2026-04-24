import Image from "next/image";
import { ArrowUpRight } from "lucide-react";
import { PrimaryButton, SecondaryButton } from "@site/src/components/Buttons";
import HeroAnnouncement from "@site/src/components/HeroAnnouncement";
import styles from "./HomeSection.module.scss";

type Brand = {
  name: string;
  slug: string;
};

const BRANDS: Brand[] = [
  // Row 1 — required anchors (LEGO col 3, Uber col 5, Google and OpenAI split)
  { name: "Google", slug: "google" },
  { name: "Uber", slug: "uber" },
  { name: "OpenAI", slug: "openai" },
  { name: "LEGO", slug: "lego" },
  { name: "Visa", slug: "visa" },
  // Row 2 — blue / red / silver / orange / red-yellow
  { name: "Toyota", slug: "toyota" },
  { name: "Adobe", slug: "adobe" },
  { name: "Walmart", slug: "walmart" },
  { name: "Mastercard", slug: "mastercard" },
  { name: "AWS", slug: "aws" },
  // Row 3 — mono / yellow-dark / green / blue-yellow / multi
  { name: "Samsung", slug: "samsung" },
  { name: "EY", slug: "ey" },
  { name: "Mercedes-Benz", slug: "benz" },
  { name: "NVIDIA", slug: "nvidia" },
  { name: "Microsoft", slug: "microsoft" },
  // Row 4 — blue / red / blue / red / teal (alternating)
  { name: "Bosch", slug: "bosch" },
  { name: "Pfizer", slug: "pfizer" },
  { name: "AXA", slug: "axa" },
  { name: "Siemens", slug: "siemens" },
  { name: "CVS Health", slug: "cvs-health" },
];

const BANNER_ITEMS = [
  "Over 100 million daily evals",
  "Used by 150K+ developers",
  "Adopted by > 50% of Fortune 500s",
];

const HomeHeroSection: React.FC = () => {
  return (
    <section className={styles.hero}>
      <div className={styles.main}>
        {/* <HeroAnnouncement
          href="/blog/deepeval-got-a-new-look"
          label="Read the DeepEval Got a New Look announcement"
        >
          DeepEval just got a new look
        </HeroAnnouncement> */}
        <h1 className={styles.title}>The LLM Evaluation Framework</h1>

        <p className={styles.description}>
          Used by some of the world&apos;s leading AI companies, DeepEval
          enables teams to build reliable evaluation pipelines to test any AI
          system.
        </p>

        <div className={styles.actions}>
          <PrimaryButton
            href="/docs/getting-started"
            endIcon={<ArrowUpRight aria-hidden />}
          >
            Get Started
          </PrimaryButton>
          <SecondaryButton href="/guides/guides-ai-agent-evaluation">
            Explore Guides
          </SecondaryButton>
        </div>
      </div>
      <div className={styles.banner} aria-label="DeepEval by the numbers">
        <div className={styles.bannerTrack}>
          {[...BANNER_ITEMS, ...BANNER_ITEMS].map((item, i) => (
            <span
              key={i}
              className={styles.bannerItem}
              aria-hidden={i >= BANNER_ITEMS.length}
            >
              {item}
            </span>
          ))}
        </div>
      </div>
      <div className={styles.logoGrid} aria-label="Companies using DeepEval">
        {BRANDS.map((brand) => (
          <div key={brand.slug} className={styles.cell}>
            <Image
              src={`/icons/companies/${brand.slug}.svg`}
              alt={brand.name}
              width={120}
              height={40}
              className={styles.logo}
            />
          </div>
        ))}
      </div>
    </section>
  );
};

export default HomeHeroSection;
