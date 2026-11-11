import { providerInitials } from "@/lib/benchmarks/format";
import styles from "./ProviderMark.module.scss";

type ProviderMarkProps = {
  provider: string;
  size?: "sm" | "md" | "lg";
};

type ProviderIcon = {
  file: string;
  // Single-colour marks are drawn in the text colour so they stay visible
  // in both themes.
  mono?: boolean;
  // Replaces `file` in dark mode, for colour marks with dark parts.
  darkFile?: string;
};

// Provider name to a file in `public/icons/providers`. Providers without an
// entry fall back to their initials.
const PROVIDER_ICONS: Record<string, ProviderIcon> = {
  Alibaba: { file: "alibaba-color" },
  Amazon: { file: "aws-color", darkFile: "aws-color-dark" },
  Anthropic: { file: "claude-color" },
  AssemblyAI: { file: "assemblyai-color" },
  DeepSeek: { file: "deepseek-color" },
  ElevenLabs: { file: "elevenlabs", mono: true },
  Google: { file: "google-color" },
  Meta: { file: "meta-color" },
  NVIDIA: { file: "nvidia-color" },
  OpenAI: { file: "openai", mono: true },
  xAI: { file: "xai", mono: true },
};

const src = (file: string) => `/icons/providers/${file}.svg`;

// The logo itself, sized by its parent.
const ProviderGlyph: React.FC<{ icon: ProviderIcon }> = ({ icon }) =>
  icon.mono ? (
    <span
      className={styles.mono}
      style={{ ["--icon" as string]: `url(${src(icon.file)})` }}
    />
  ) : (
    <>
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img
        src={src(icon.file)}
        alt=""
        className={styles.image}
        data-theme={icon.darkFile ? "light" : undefined}
      />
      {icon.darkFile && (
        // eslint-disable-next-line @next/next/no-img-element
        <img
          src={src(icon.darkFile)}
          alt=""
          className={styles.image}
          data-theme="dark"
        />
      )}
    </>
  );

const ProviderMark: React.FC<ProviderMarkProps> = ({ provider, size = "md" }) => {
  const icon = PROVIDER_ICONS[provider];

  if (!icon) {
    return (
      <span className={styles.initials} data-size={size} title={provider} aria-hidden>
        {providerInitials(provider)}
      </span>
    );
  }

  return (
    <span className={styles.mark} data-size={size} title={provider} aria-hidden>
      <ProviderGlyph icon={icon} />
    </span>
  );
};

// Oversized version of the logo for use as a backdrop. Renders nothing for
// providers without a logo.
export const ProviderBackdrop: React.FC<{
  provider: string;
  className?: string;
}> = ({ provider, className }) => {
  const icon = PROVIDER_ICONS[provider];
  if (!icon) return null;

  return (
    <span
      className={[styles.backdrop, className].filter(Boolean).join(" ")}
      aria-hidden
    >
      <ProviderGlyph icon={icon} />
    </span>
  );
};

export default ProviderMark;
