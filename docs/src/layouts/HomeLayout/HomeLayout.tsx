"use client";

import { type ReactNode, useEffect, useRef } from "react";
import styles from "./HomeLayout.module.scss";
import Footer from "../Footer";

type HomeLayoutProps = {
  leftContent: ReactNode;
  rightContent: ReactNode;
};

const HomeLayout: React.FC<HomeLayoutProps> = ({
  leftContent,
  rightContent,
}) => {
  const stageRef = useRef<HTMLDivElement>(null);
  const frameRef = useRef<HTMLDivElement>(null);
  const rightPaneRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const stage = stageRef.current;
    const frame = frameRef.current;
    const rightPane = rightPaneRef.current;

    if (!stage || !frame || !rightPane) {
      return;
    }

    const desktopQuery = window.matchMedia("(min-width: 1024px)");
    let resizeFrame = 0;
    let scrollFrame = 0;

    const getHeaderOffset = () => {
      const headerHeight = Number.parseFloat(
        window.getComputedStyle(frame).getPropertyValue("--home-header-height"),
      );

      return Number.isFinite(headerHeight) ? headerHeight : 0;
    };

    const syncStageHeight = () => {
      if (!desktopQuery.matches) {
        stage.style.removeProperty("--right-pane-scroll-range");
        rightPane.scrollTop = 0;
        return;
      }

      const scrollRange = Math.max(
        0,
        rightPane.scrollHeight - rightPane.clientHeight,
      );

      stage.style.setProperty("--right-pane-scroll-range", `${scrollRange}px`);
    };

    const syncRightPaneScroll = () => {
      if (!desktopQuery.matches) {
        rightPane.scrollTop = 0;
        return;
      }

      const scrollRange = Math.max(
        0,
        rightPane.scrollHeight - rightPane.clientHeight,
      );
      const progress = getHeaderOffset() - stage.getBoundingClientRect().top;
      rightPane.scrollTop = Math.min(Math.max(progress, 0), scrollRange);
    };

    const requestStageSync = () => {
      window.cancelAnimationFrame(resizeFrame);
      resizeFrame = window.requestAnimationFrame(() => {
        syncStageHeight();
        syncRightPaneScroll();
      });
    };

    const requestScrollSync = () => {
      window.cancelAnimationFrame(scrollFrame);
      scrollFrame = window.requestAnimationFrame(syncRightPaneScroll);
    };

    const resizeObserver = new ResizeObserver(requestStageSync);
    resizeObserver.observe(frame);
    resizeObserver.observe(rightPane);

    const rightPaneContent = rightPane.firstElementChild;
    if (rightPaneContent instanceof HTMLElement) {
      resizeObserver.observe(rightPaneContent);
    }

    desktopQuery.addEventListener("change", requestStageSync);
    window.addEventListener("resize", requestStageSync);
    window.addEventListener("scroll", requestScrollSync, { passive: true });

    requestStageSync();

    return () => {
      window.cancelAnimationFrame(resizeFrame);
      window.cancelAnimationFrame(scrollFrame);
      resizeObserver.disconnect();
      desktopQuery.removeEventListener("change", requestStageSync);
      window.removeEventListener("resize", requestStageSync);
      window.removeEventListener("scroll", requestScrollSync);
    };
  }, []);

  return (
    <section className={styles.layout}>
      <div ref={stageRef} className={styles.stage}>
        <div
          aria-hidden="true"
          className={`${styles.stageRightRail} paper-grid-surface`}
        />
        <div ref={frameRef} className={styles.frame}>
          <aside className={styles.leftPane}>{leftContent}</aside>
          <div ref={rightPaneRef} className={styles.rightPane}>
            <div className={styles.rightPaneContent}>{rightContent}</div>
          </div>
        </div>
      </div>
      <Footer />
    </section>
  );
};


export default HomeLayout;
