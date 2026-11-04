"use client";

import { useEffect, useId, useRef, useState } from "react";
import Image from "next/image";
import Link from "next/link";
import { Check, ChevronDown, Copy, SquareTerminal } from "lucide-react";
import { OpenAIMark } from "@site/src/components/BrandMarks";
import styles from "./BuildWithAgents.module.scss";

type CommandIcon = "cli" | "python" | "typescript";

type Command = {
  label?: string;
  icon?: CommandIcon;
  command: string;
};

type Option = {
  id: string;
  name: string;
  description: string;
  commands: Command[];
};

const OPTIONS: Option[] = [
  {
    id: "wizard",
    name: "Wizard",
    description:
      "Walks your coding agent through building and running a first evaluation on your repo.",
    commands: [
      {
        label: "CLI",
        icon: "cli",
        command: "curl -fsSL https://deepeval.com/setup.sh | sh",
      },
      {
        label: "Python",
        icon: "python",
        command: "pip install -U deepeval && deepeval setup",
      },
      {
        label: "TypeScript",
        icon: "typescript",
        command: "npx deepeval setup",
      },
    ],
  },
  {
    id: "skill",
    name: "Skill",
    description:
      "Teaches Claude Code, Codex, Cursor and other agents to write, run and iterate on DeepEval evals.",
    commands: [
      { command: 'npx skills add confident-ai/deepeval --skill "deepeval"' },
    ],
  },
];

function LabelIcon({ icon }: { icon: CommandIcon }) {
  if (icon === "cli") {
    return <SquareTerminal aria-hidden className={styles.labelIcon} />;
  }
  return (
    <Image
      src={`/icons/${icon}.svg`}
      alt=""
      width={14}
      height={14}
      className={styles.labelIcon}
    />
  );
}

function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    if (!copied) return;
    const timeout = setTimeout(() => setCopied(false), 1500);
    return () => clearTimeout(timeout);
  }, [copied]);

  return (
    <button
      type="button"
      className={styles.copy}
      aria-label={copied ? "Copied" : "Copy command"}
      onClick={() => {
        navigator.clipboard
          ?.writeText(text)
          .then(() => setCopied(true))
          .catch(() => {});
      }}
    >
      {copied ? <Check aria-hidden /> : <Copy aria-hidden />}
    </button>
  );
}

const BuildWithAgents: React.FC = () => {
  const [open, setOpen] = useState(false);
  const [active, setActive] = useState(OPTIONS[0].id);
  const rootRef = useRef<HTMLDivElement>(null);
  const panelId = useId();
  const option = OPTIONS.find((o) => o.id === active) ?? OPTIONS[0];

  useEffect(() => {
    if (!open) return;
    const onPointerDown = (event: PointerEvent) => {
      if (!rootRef.current?.contains(event.target as Node)) setOpen(false);
    };
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") setOpen(false);
    };
    document.addEventListener("pointerdown", onPointerDown);
    document.addEventListener("keydown", onKeyDown);
    return () => {
      document.removeEventListener("pointerdown", onPointerDown);
      document.removeEventListener("keydown", onKeyDown);
    };
  }, [open]);

  return (
    <div ref={rootRef} className={styles.root}>
      <button
        type="button"
        className={styles.trigger}
        aria-expanded={open}
        aria-controls={panelId}
        onClick={() => setOpen((value) => !value)}
      >
        <span className={styles.marks} aria-hidden>
          <Image
            src="/icons/integrations/claude.svg"
            alt=""
            width={14}
            height={14}
          />
          <OpenAIMark className={styles.mark} />
          <SquareTerminal className={styles.mark} />
        </span>
        Build with agents
        <ChevronDown
          aria-hidden
          className={styles.chevron}
          data-open={open || undefined}
        />
      </button>

      {open && (
        <div id={panelId} className={styles.panel}>
          <div className={styles.tabs} role="tablist">
            {OPTIONS.map((o) => (
              <button
                key={o.id}
                type="button"
                role="tab"
                aria-selected={o.id === active}
                className={styles.tab}
                onClick={() => setActive(o.id)}
              >
                {o.name}
              </button>
            ))}
          </div>

          <div role="tabpanel" className={styles.body}>
            <p className={styles.description}>{option.description}</p>
            {option.commands.map((c) => (
              <div key={c.command} className={styles.command}>
                {c.label && (
                  <span className={styles.label}>
                    {c.icon && <LabelIcon icon={c.icon} />}
                    {c.label}
                  </span>
                )}
                <code>{c.command}</code>
                <CopyButton text={c.command} />
              </div>
            ))}
          </div>

          <Link
            href="/docs/vibe-coder-quickstart#universal-starter-prompt"
            className={styles.footer}
          >
            Prefer a prompt? Use the starter prompt →
          </Link>
        </div>
      )}
    </div>
  );
};

export default BuildWithAgents;
