import type { ComponentType, ReactNode, SVGProps } from "react";
import { Check, X } from "lucide-react";
import {
  CartesiaMark,
  DeepgramMark,
  ElevenLabsMark,
  OpenAIMark,
} from "@site/src/components/BrandMarks";

type Provider = "openai" | "elevenlabs" | "cartesia" | "deepgram";

/** `true` and `false` render as a tick and a cross; a string is shown as written. */
type Support = boolean | string;

type Row = { setting: string } & Record<Provider, Support>;

interface TtsSupportTableProps {
  rows: Row[];
}

const PROVIDERS: {
  key: Provider;
  name: string;
  href: string;
  Mark: ComponentType<SVGProps<SVGSVGElement>>;
}[] = [
  { key: "openai", name: "OpenAI", href: "/integrations/tts/openai", Mark: OpenAIMark },
  { key: "elevenlabs", name: "ElevenLabs", href: "/integrations/tts/elevenlabs", Mark: ElevenLabsMark },
  { key: "cartesia", name: "Cartesia", href: "/integrations/tts/cartesia", Mark: CartesiaMark },
  { key: "deepgram", name: "Deepgram", href: "/integrations/tts/deepgram", Mark: DeepgramMark },
];

function cell(support: Support): ReactNode {
  if (support === true) return <Check aria-label="Supported" className="mx-auto size-4 text-emerald-500" strokeWidth={3} />;
  if (support === false) return <X aria-label="Not supported" className="mx-auto size-4 text-fd-muted-foreground" strokeWidth={3} />;
  return <span className="text-xs text-fd-muted-foreground">{support}</span>;
}

/** Which persona settings each TTS model can act on. */
const TtsSupportTable: React.FC<TtsSupportTableProps> = ({ rows }) => (
  <div className="not-prose my-4 overflow-x-auto rounded-xl border border-fd-border">
    <table className="w-full border-collapse text-sm">
      <thead>
        <tr className="border-b border-fd-border bg-fd-muted">
          <th className="px-3 py-2 text-left font-semibold">Setting</th>
          {PROVIDERS.map(({ key, name, href, Mark }) => (
            <th key={key} className="px-3 py-2 font-semibold">
              <a href={href} className="flex flex-col items-center gap-1 !text-fd-foreground !no-underline hover:opacity-70">
                <Mark className="size-6" aria-hidden />
                <span className="text-xs">{name}</span>
              </a>
            </th>
          ))}
        </tr>
      </thead>
      <tbody>
        {rows.map((row) => (
          <tr key={row.setting} className="border-b border-fd-border last:border-b-0">
            <td className="px-3 py-2">
              <code className="rounded bg-fd-muted px-1.5 py-0.5 font-mono text-[13px]">{row.setting}</code>
            </td>
            {PROVIDERS.map(({ key }) => (
              <td key={key} className="px-3 py-2 text-center">
                {cell(row[key])}
              </td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  </div>
);

export default TtsSupportTable;
