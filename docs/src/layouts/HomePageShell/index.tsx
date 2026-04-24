"use client";

import type { ReactNode } from "react";
import SiteTopNav from "@/src/layouts/SiteTopNav";
import styles from "./HomePageShell.module.scss";

type HomePageShellProps = {
  children: ReactNode;
};

const HomePageShell: React.FC<HomePageShellProps> = ({ children }) => {
  return (
    <div className={styles.shell}>
      <SiteTopNav variant="home" dataTransparent="false" />

      <main className={styles.main}>{children}</main>
    </div>
  );
};


export default HomePageShell;
