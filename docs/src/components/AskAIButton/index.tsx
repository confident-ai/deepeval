import type { ReactNode } from "react";
import { Sparkles } from "lucide-react";
import { PrimaryButton } from "@/src/components/Buttons";
import { kapaConfig } from "@/lib/shared";

/**
 * "Ask AI" call-to-action that wraps the site's {@link PrimaryButton}
 * so visual parity with every other CTA is automatic (same padding,
 * radius, hover treatment, Tailwind theme tokens).
 *
 * Trigger wiring is entirely declarative: the Kapa widget is loaded
 * once in `app/layout.tsx` with
 * `data-modal-override-selector=".{kapaConfig.triggerClass}"`, so any
 * click on an element carrying that class opens Kapa's "Ask DeepEval"
 * modal. This component just applies the class — there is no onClick
 * handler to get stale, lose closures, or race the script load.
 *
 * Usage is intentionally minimal:
 *   `<AskAIButton />`                 → default "Ask AI" label
 *   `<AskAIButton label="Ask DeepEval" />`
 */

type AskAIButtonProps = {
  label?: ReactNode;
};

const AskAIButton: React.FC<AskAIButtonProps> = ({ label = "Ask AI" }) => {
  return (
    <span className={kapaConfig.triggerClass}>
      <PrimaryButton
        type="button"
        startIcon={<Sparkles aria-hidden="true" />}
        aria-label={typeof label === "string" ? label : "Ask AI"}
        shortkey="K"
      >
        {label}
      </PrimaryButton>
    </span>
  );
};


export default AskAIButton;
