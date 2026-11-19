"use client";

import type { ReactNode } from "react";
import { MeetupBanner } from "@/lib/announcement";
import SiteTopNav from "@/src/layouts/SiteTopNav";
import styles from "./HomePageShell.module.scss";

type HomePageShellProps = {
  children: ReactNode;
};

const HomePageShell: React.FC<HomePageShellProps> = ({ children }) => {
  return (
    <div className={styles.shell}>
      <MeetupBanner changeLayout={false} />
      <SiteTopNav variant="home" dataTransparent="false" />

      <main className={styles.main}>{children}</main>
    </div>
  );
};


export default HomePageShell;
