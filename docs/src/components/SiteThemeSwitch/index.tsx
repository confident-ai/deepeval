"use client";

import { ThemeSwitch } from "fumadocs-ui/layouts/shared/slots/theme-switch";
import styles from "./SiteThemeSwitch.module.scss";

const SiteThemeSwitch: React.FC = () => {
  return <ThemeSwitch className={styles.switch} />;
};


export default SiteThemeSwitch;
