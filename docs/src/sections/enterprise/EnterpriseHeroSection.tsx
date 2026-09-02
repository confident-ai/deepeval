import { ArrowUpRight } from "lucide-react";
import { externalRelForOutboundHref } from "@/src/utils/outbound-link-rel";
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
    name: "LEGO",
    slug: "lego",
    src: "/icons/companies/lego.svg",
  },
  { name: "Panasonic", slug: "panasonic" },
  { name: "Finom", slug: "finom" },
  { name: "Siemens", slug: "siemens" },
  { name: "Toshiba", slug: "toshiba" },
  { name: "BCG", slug: "bcg" },
  {
    name: "Epic Games",
    slug: "epic-games",
  },
  {
    name: "Phreesia",
    slug: "phreesia",
  },
];

const DEFAULT_LOGO_GRID_LABEL =
  "Trusted by teams that took evals to production.";

const BOOK_DEMO_HREF =
  "https://calendly.com/d/d3m7-g99-8ct/deepeval-enterprise-intro-call";

const EnterpriseHeroSection: React.FC<EnterpriseHeroSectionProps> = ({
  logoItems = DEFAULT_LOGO_ITEMS,
  logoGridLabel = DEFAULT_LOGO_GRID_LABEL,
}) => {
  return (
    <section className={styles.hero}>
      <div className={styles.main}>
        <h1 className={styles.title}>
          Standardize evals across your organization with DeepEval enterprise.
        </h1>

        <p className={styles.description}>
          AI governance and eval workflows for your engineers, PMs, and QA to
          use together — not just developers in a terminal.
        </p>

        <div className={styles.actions}>
          <PrimaryButton
            href={BOOK_DEMO_HREF}
            target="_blank"
            rel={externalRelForOutboundHref(BOOK_DEMO_HREF)}
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
                src={brand.src ?? `/icons/brand-icons/${brand.slug}.svg`}
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
