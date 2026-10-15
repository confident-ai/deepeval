import { icons as lucideIcons, type LucideIcon } from "lucide-react";
import * as simpleIcons from "simple-icons";
import type { SimpleIcon } from "simple-icons";
import { CHART_COLORS, type ChartColor } from "@/lib/charts/schema";
import { loadLibrary, resolveMedia } from "@/lib/media/library.mjs";
import { EXTRA_BRANDS } from "./board-brands";

export const ICON_SOURCES = ["brand", "icon", "media"] as const;
/** `<source>:<name>`, optionally `@<chart color>` to draw it in that color, e.g. `icon:bot@violet`. */
export const ICON_REF = /^(brand|icon|media):([^\s@]+)(?:@(\S+))?$/;

/** The chart color an icon is drawn in, when its reference names one. */
export const iconTint = (ref: string): ChartColor | undefined => {
  const color = ICON_REF.exec(ref)?.[3];
  return CHART_COLORS.find(c => c === color);
};

export type ResolvedIcon =
  | { source: "brand"; title: string; path: string; color: string | null }
  | { source: "icon"; title: string; Component: LucideIcon }
  | { source: "media"; title: string; src: string; width: number; height: number };

const BRANDS = new Map<string, { title: string; hex: string; path: string }>([
  ...Object.values(simpleIcons as Record<string, SimpleIcon>)
    .filter(icon => typeof icon === "object" && "slug" in icon)
    .map(icon => [icon.slug, icon] as const),
  ...Object.entries(EXTRA_BRANDS),
]);

const pascal = (name: string) =>
  name
    .split("-")
    .map(part => part.charAt(0).toUpperCase() + part.slice(1))
    .join("");

const lucide = (name: string): LucideIcon | undefined =>
  /^[a-z0-9]+(?:-[a-z0-9]+)*$/.test(name) && Object.hasOwn(lucideIcons, pascal(name))
    ? lucideIcons[pascal(name) as keyof typeof lucideIcons]
    : undefined;

const luminance = (hex: string) => {
  const [r, g, b] = [0, 2, 4].map(i => {
    const c = parseInt(hex.slice(i, i + 2), 16) / 255;
    return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
  });
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
};

const brandColor = (hex: string) => {
  const l = luminance(hex);
  return l < 0.04 || l > 0.85 ? null : `#${hex}`;
};

export const resolveIcon = (ref: string): ResolvedIcon | null => {
  const match = ICON_REF.exec(ref);
  if (!match) return null;
  const [, source, name] = match;
  if (source === "brand") {
    const icon = BRANDS.get(name);
    return icon ? { source, title: icon.title, path: icon.path, color: brandColor(icon.hex) } : null;
  }
  if (source === "icon") {
    const Component = lucide(name);
    return Component ? { source, title: name.replace(/-/g, " "), Component } : null;
  }
  const entry = loadLibrary().get(name);
  return entry ? { source: "media", title: entry.title, ...resolveMedia(name) } : null;
};

export const iconProblem = (ref: string): string | null => {
  const match = ICON_REF.exec(ref);
  if (!match) return `icon "${ref}" needs a source: "brand:<slug>" (Simple Icons), "icon:<name>" (Lucide) or "media:<id>" (media library)`;
  const [, source, name, color] = match;
  if (color && !iconTint(ref)) return `unknown color "${color}" in icon "${ref}": use ${CHART_COLORS.map(c => `"@${c}"`).join(", ")}`;
  if (color && source === "media") return `media images keep their own colors; drop "@${color}" from "${ref}"`;
  if (resolveIcon(ref)) return null;
  if (source === "brand") {
    return `unknown brand "${name}": use a slug from simpleicons.org (or ${Object.keys(EXTRA_BRANDS).join(", ")}), or add the logo to docs/media/ and use "media:<id>"`;
  }
  if (source === "icon") return `unknown Lucide icon "${name}": use a kebab-case name from lucide.dev/icons`;
  return `unknown media id "${name}": add the image to docs/media/ first`;
};
