"use client";

import { useRef, type ComponentProps, type ReactNode } from "react";
import Link from "next/link";
import Hotkey, { type HotkeyConfig } from "@/src/components/Hotkey";
import styles from "./Buttons.module.scss";

type CommonButtonProps = {
  children: ReactNode;
  startIcon?: ReactNode;
  endIcon?: ReactNode;
};

type PrimaryExtras = {
  shortkey?: string;
};

type RenderContentProps = CommonButtonProps &
  PrimaryExtras & {
    hotkey?: HotkeyConfig;
  };

type LinkButtonProps = CommonButtonProps &
  Omit<ComponentProps<typeof Link>, "className" | "children"> & {
    href: ComponentProps<typeof Link>["href"];
  };

type NativeButtonProps = CommonButtonProps &
  Omit<ComponentProps<"button">, "className" | "children"> & {
    href?: undefined;
  };

type ButtonProps = LinkButtonProps | NativeButtonProps;
type PrimaryButtonProps =
  | (LinkButtonProps & PrimaryExtras)
  | (NativeButtonProps & PrimaryExtras);

function renderContent({
  children,
  startIcon,
  endIcon,
  shortkey,
  hotkey,
}: RenderContentProps) {
  return (
    <>
      {startIcon && <span className={styles.startIcon}>{startIcon}</span>}
      {children}
      {endIcon && <span className={styles.endIcon}>{endIcon}</span>}
      {shortkey && hotkey && <Hotkey hotkey={hotkey} />}
    </>
  );
}

export const PrimaryButton: React.FC<PrimaryButtonProps> = (props) => {
  const { shortkey, ...rest } = props;
  const linkRef = useRef<HTMLAnchorElement>(null);
  const buttonRef = useRef<HTMLButtonElement>(null);

  if ("href" in rest && rest.href !== undefined) {
    const { children, startIcon, endIcon, ...linkProps } = rest;
    return (
      <Link
        {...linkProps}
        ref={linkRef}
        className={styles.primary}
        data-button
        data-callout
      >
        {renderContent({
          children,
          startIcon,
          endIcon,
          shortkey,
          hotkey: shortkey
            ? {
                key: shortkey,
                action: () => linkRef.current?.click(),
              }
            : undefined,
        })}
      </Link>
    );
  }

  const {
    children,
    startIcon,
    endIcon,
    type = "button",
    ...buttonProps
  } = rest;
  return (
    <button
      {...buttonProps}
      ref={buttonRef}
      type={type}
      className={styles.primary}
      data-button
      data-callout
    >
      {renderContent({
        children,
        startIcon,
        endIcon,
        shortkey,
        hotkey: shortkey
          ? {
              key: shortkey,
              action: () => buttonRef.current?.click(),
            }
          : undefined,
      })}
    </button>
  );
};

export const SecondaryButton: React.FC<ButtonProps> = (props) => {
  const { ...rest } = props;

  if ("href" in rest && rest.href !== undefined) {
    const { children, startIcon, endIcon, ...linkProps } = rest;
    return (
      <Link {...linkProps} className={styles.secondary} data-button data-callout>
        {renderContent({ children, startIcon, endIcon })}
      </Link>
    );
  }

  const {
    children,
    startIcon,
    endIcon,
    type = "button",
    ...buttonProps
  } = rest;
  return (
    <button
      {...buttonProps}
      type={type}
      className={styles.secondary}
      data-button
      data-callout
    >
      {renderContent({ children, startIcon, endIcon })}
    </button>
  );
};
