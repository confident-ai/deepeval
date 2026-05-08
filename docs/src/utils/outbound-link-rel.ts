export const externalRelForOutboundHref = (href: string): string => {
  try {
    const host = new URL(href).hostname.toLowerCase();
    if (host === "confident-ai.com" || host.endsWith(".confident-ai.com")) {
      return "noopener";
    }
  } catch {}
  return "noopener noreferrer";
};
