"use client";

import { type ComponentProps, type ComponentType } from "react";
import Link from "next/link";
import { buttonVariants } from "fumadocs-ui/components/ui/button";
import { Sidebar } from "lucide-react";
import { twMerge } from "tailwind-merge";
import DiscordButton from "@/src/components/DiscordButton";
import GithubCtaButton from "@/src/components/GithubCtaButton";
import SiteThemeSwitch from "@/src/components/SiteThemeSwitch";
import { appName } from "@/lib/shared";
import { navLinks } from "@/lib/layout.shared";
import AskAIButton from "@/src/components/AskAIButton";
import HomeOverflowNav from "@/src/layouts/HomeOverflowNav";
import NavLinks from "@/src/layouts/NavLinks";
import NavMenu from "@/src/layouts/NavMenu";
import styles from "./SiteTopNav.module.scss";

type NavTitleComponent = ComponentType<{ className?: string }>;
type ThemeSwitchComponent = ComponentType<{ className?: string }>;
type CollapseTriggerComponent = ComponentType<ComponentProps<"button">>;

type SiteTopNavProps = {
  variant: "docs" | "home";
  dataTransparent?: boolean | "false" | "true";
  navTitle?: NavTitleComponent | false;
  themeSwitch?: ThemeSwitchComponent | false;
  collapseTrigger?: CollapseTriggerComponent | false;
  headerProps?: ComponentProps<"header">;
};

const SiteTopNav: React.FC<SiteTopNavProps> = ({
  variant,
  dataTransparent,
  navTitle: NavTitle = false,
  themeSwitch: ThemeSwitchSlot = false,
  collapseTrigger: CollapseTrigger = false,
  headerProps,
}) => {
  const {
    className: headerClassName,
    children: _ignoredChildren,
    ...restHeaderProps
  } = headerProps ?? {};

  void _ignoredChildren;

  if (variant === "docs") {
    const headerClassNameCombined = twMerge(
      "sticky [grid-area:header] top-(--fd-docs-row-1) z-10 backdrop-blur-sm transition-colors data-[transparent=false]:bg-fd-background/80 layout:[--fd-header-height:--spacing(14)]",
      styles.headerDocs,
      headerClassName
    );

    const themeSwitchNode = ThemeSwitchSlot ? (
      <ThemeSwitchSlot className={styles.utilityThemeSwitch} />
    ) : null;

    return (
      <header
        id="nd-subnav"
        data-transparent={dataTransparent}
        className={headerClassNameCombined}
        {...restHeaderProps}
      >
        <div data-header-body="" className={styles.row}>
          <div className={styles.logoCell}>
            {NavTitle ? <NavTitle className={styles.docsLogo} /> : null}
          </div>

          <div className={styles.mainCell}>
            <div className={styles.mainNavLinks}>
              <NavLinks items={navLinks} />
            </div>
            <div className={styles.mainMenuTrigger}>
              <NavMenu items={navLinks} />
            </div>
          </div>

          <div className={styles.utilsCell}>
            <div className={styles.utilsDesktop}>
              <AskAIButton />
              {themeSwitchNode}
              {CollapseTrigger ? (
                <CollapseTrigger
                  className={twMerge(
                    buttonVariants({ size: "icon-sm", color: "secondary" }),
                    "text-fd-muted-foreground rounded-none"
                  )}
                >
                  <Sidebar />
                </CollapseTrigger>
              ) : null}
            </div>
            <div className={styles.utilsMobile}>
              <NavMenu items={navLinks} />
            </div>
          </div>
        </div>
      </header>
    );
  }

  const headerClassNameCombined = twMerge(
    styles.headerHome,
    headerClassName
  );

  return (
    <header
      id="nd-subnav"
      data-transparent={dataTransparent}
      className={headerClassNameCombined}
      {...restHeaderProps}
    >
      <div className={styles.homeFrame}>
        <div
          data-header-body=""
          className={`${styles.row} ${styles.homeRow}`}
        >
          <div className={styles.homeLogoCell}>
            <Link href="/" className={styles.brandLink} aria-label={appName}>
              <span
                className={`${styles.wordmark} ${styles.wordmarkHome}`}
                role="img"
                aria-label={appName}
              />
            </Link>
          </div>

          <div className={styles.homeMainCell}>
            <div className={styles.homeNavDesktop}>
              <HomeOverflowNav items={navLinks} />
            </div>
          </div>

          <div className={styles.homeUtilsCell}>
            <div className={styles.homeUtilities}>
              <div className={styles.homeDiscordCta}>
                <DiscordButton layout="inline" />
              </div>
              <div className={styles.homeGithubCta}>
                <GithubCtaButton layout="inline" tone="secondary" />
              </div>
              <SiteThemeSwitch />
              <NavMenu items={navLinks} />
            </div>
          </div>
        </div>
      </div>
    </header>
  );
};


export default SiteTopNav;
