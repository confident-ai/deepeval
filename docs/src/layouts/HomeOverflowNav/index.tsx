"use client";

import * as Popover from "@radix-ui/react-popover";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { ChevronDown } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";
import { twMerge } from "tailwind-merge";
import {
  NavLinkItem,
  isNavLinkActive,
  navLinkClassName,
  navLinksListClassName,
  type NavLink,
} from "@/src/layouts/NavLinks";
import styles from "./HomeOverflowNav.module.scss";

type HomeOverflowNavProps = {
  items: NavLink[];
};

const HomeOverflowNav: React.FC<HomeOverflowNavProps> = ({ items }) => {
  const pathname = usePathname();
  const containerRef = useRef<HTMLDivElement>(null);
  const measureListRef = useRef<HTMLUListElement>(null);
  const moreMeasureRef = useRef<HTMLLIElement>(null);
  const itemMeasureRefs = useRef<Array<HTMLLIElement | null>>([]);
  const [visibleCount, setVisibleCount] = useState(items.length);
  const [ready, setReady] = useState(false);

  const recomputeVisibleCount = useCallback(() => {
    const containerWidth = containerRef.current?.getBoundingClientRect().width ?? 0;
    const moreWidth = moreMeasureRef.current?.getBoundingClientRect().width ?? 0;
    const gapValue =
      measureListRef.current
        ? parseFloat(
            getComputedStyle(measureListRef.current).columnGap ||
              getComputedStyle(measureListRef.current).gap ||
              "0"
          )
        : 0;

    const widths = items.map(
      (_, index) =>
        itemMeasureRefs.current[index]?.getBoundingClientRect().width ?? 0
    );

    if (!containerWidth || widths.some((width) => width === 0)) {
      return;
    }

    const prefixWidth = (count: number) =>
      widths.slice(0, count).reduce((sum, width) => sum + width, 0) +
      Math.max(0, count - 1) * gapValue;

    let nextVisibleCount = 0;

    for (let count = items.length; count >= 0; count -= 1) {
      const hasOverflow = count < items.length;
      const totalWidth =
        prefixWidth(count) +
        (hasOverflow ? moreWidth + (count > 0 ? gapValue : 0) : 0);

      if (totalWidth <= containerWidth) {
        nextVisibleCount = count;
        break;
      }
    }

    setVisibleCount((current) =>
      current === nextVisibleCount ? current : nextVisibleCount
    );
    setReady(true);
  }, [items]);

  useEffect(() => {
    const frame = window.requestAnimationFrame(recomputeVisibleCount);
    const observer = new ResizeObserver(recomputeVisibleCount);
    const fontReady = document.fonts?.ready;

    if (containerRef.current) observer.observe(containerRef.current);
    if (measureListRef.current) observer.observe(measureListRef.current);
    if (moreMeasureRef.current) observer.observe(moreMeasureRef.current);
    itemMeasureRefs.current.forEach((node) => {
      if (node) observer.observe(node);
    });

    if (fontReady) {
      void fontReady.then(recomputeVisibleCount);
    }

    return () => {
      window.cancelAnimationFrame(frame);
      observer.disconnect();
    };
  }, [recomputeVisibleCount]);

  const visibleItems = items.slice(0, visibleCount);
  const overflowItems = items.slice(visibleCount);
  const hasActiveOverflow = overflowItems.some((item) =>
    isNavLinkActive(pathname, item)
  );

  return (
    <div
      ref={containerRef}
      className={styles.root}
      data-ready={ready}
    >
      <div className={styles.visible}>
        <ul className={twMerge(navLinksListClassName, styles.inlineList)}>
          {visibleItems.map((item) => (
            <NavLinkItem key={item.url} item={item} pathname={pathname} />
          ))}
          {overflowItems.length > 0 ? (
            <li>
              <Popover.Root>
                <Popover.Trigger asChild>
                  <button
                    type="button"
                    data-active={hasActiveOverflow}
                    className={twMerge(navLinkClassName, styles.moreTrigger)}
                    aria-label={`Show ${overflowItems.length} more navigation items`}
                  >
                    <span>More</span>
                    <ChevronDown className={styles.moreChevron} />
                  </button>
                </Popover.Trigger>
                <Popover.Portal>
                  <Popover.Content
                    align="start"
                    sideOffset={8}
                    collisionPadding={8}
                    className={styles.content}
                  >
                    <ul className={styles.menuList}>
                      {overflowItems.map((item) => {
                        const active = isNavLinkActive(pathname, item);

                        return (
                          <li key={item.url}>
                            <Popover.Close asChild>
                              <Link
                                href={item.url}
                                data-active={active}
                                className={styles.menuItem}
                              >
                                <span className={styles.menuIcon}>
                                  {item.icon}
                                </span>
                                <span>{item.text}</span>
                              </Link>
                            </Popover.Close>
                          </li>
                        );
                      })}
                    </ul>
                  </Popover.Content>
                </Popover.Portal>
              </Popover.Root>
            </li>
          ) : null}
        </ul>
      </div>

      <div className={styles.measure} aria-hidden="true">
        <ul
          ref={measureListRef}
          className={twMerge(navLinksListClassName, styles.measureList)}
        >
          {items.map((item, index) => (
            <li
              key={item.url}
              ref={(node) => {
                itemMeasureRefs.current[index] = node;
              }}
            >
              <button
                type="button"
                data-active={isNavLinkActive(pathname, item)}
                className={twMerge(navLinkClassName, styles.measureLink)}
              >
                {item.icon}
                {item.text}
              </button>
            </li>
          ))}
          <li ref={moreMeasureRef}>
            <button
              type="button"
              className={twMerge(navLinkClassName, styles.moreTrigger)}
            >
              <span>More</span>
              <ChevronDown className={styles.moreChevron} />
            </button>
          </li>
        </ul>
      </div>
    </div>
  );
};


export default HomeOverflowNav;
