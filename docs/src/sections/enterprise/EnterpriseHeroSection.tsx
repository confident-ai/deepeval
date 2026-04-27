import { ArrowUpRight } from "lucide-react";
import { PrimaryButton } from "@site/src/components/Buttons";
import styles from "@site/src/sections/home/HomeSection.module.scss";
import enterpriseStyles from "./EnterpriseHeroSection.module.scss";

export type EnterpriseLogoItem = {
  name: string;
  slug: string;
  src?: string;
};

type EnterpriseHeroSectionProps = {
  logoItems?: EnterpriseLogoItem[];
  logoGridLabel?: string;
};

const DEFAULT_LOGO_ITEMS: EnterpriseLogoItem[] = [
  {
    name: "Syngenta Group",
    slug: "syngenta-group",
    src: "/icons/brand-icons/syngenta-group.svg",
  },
  { name: "Panasonic", slug: "panasonic" },
  { name: "Finom", slug: "finom" },
  { name: "Humach", slug: "humach", src: "/icons/brand-icons/humach.svg" },
  { name: "Toshiba", slug: "toshiba" },
  { name: "BCG", slug: "bcg" },
  {
    name: "Epic Games",
    slug: "epic-games",
    src: "/icons/brand-icons/epic-games.svg",
  },
  {
    name: "Phreesia",
    slug: "phreesia",
    src: "/icons/brand-icons/phreesia.svg",
  },
];

const DEFAULT_LOGO_GRID_LABEL =
  "Trusted by teams that took evals to production.";

const EnterpriseHeroSection: React.FC<EnterpriseHeroSectionProps> = ({
  logoItems = DEFAULT_LOGO_ITEMS,
  logoGridLabel = DEFAULT_LOGO_GRID_LABEL,
}) => {
  return (
    <section className={styles.hero}>
      <div className={styles.main}>
        <h1 className={styles.title}>
          Scale DeepEval with the platform built for the whole team.
        </h1>

        <p className={styles.description}>
          Production tracing, eval monitoring, and a workflow your engineers,
          PMs, and QA can use together — not just developers in a terminal.
        </p>

        <div className={styles.actions}>
          <PrimaryButton
            href="https://www.confident-ai.com/book-a-demo"
            target="_blank"
            rel="noopener noreferrer"
            data-utm-content="enterprise_hero_demo"
            endIcon={<ArrowUpRight aria-hidden />}
          >
            Book a Demo
          </PrimaryButton>
        </div>
      </div>

      <div className={enterpriseStyles.logoGridWrap}>
        <p className={enterpriseStyles.logoGridLabel}>{logoGridLabel}</p>
        <div
          className={`${styles.logoGrid} ${enterpriseStyles.logoGrid}`}
          aria-label="Companies using Confident AI"
        >
          {logoItems.map((brand) => (
            <div key={brand.slug} className={styles.cell}>
              <img
                src={
                  brand.src ??
                  `https://www.confident-ai.com/icons/brand-icons/${brand.slug}.svg`
                }
                alt={brand.name}
                className={styles.logo}
              />
            </div>
          ))}
        </div>
      </div>
    </section>
  );
};

export default EnterpriseHeroSection;
