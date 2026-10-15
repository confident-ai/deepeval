/** The markdown every frame exports: its head, the description, its data, then note and caption. */
export const frameMarkdown = (attrs: Record<string, string>, body: string): string => {
  const title = attrs.title.replace(/\*([^*]+)\*/g, "$1");
  return [
    `**${attrs.kicker ? `${attrs.kicker}: ` : ""}${title}**`,
    attrs.description ?? "",
    body,
    attrs.note ? `*${attrs.note}*` : "",
    attrs.caption ? `*${attrs.caption}*` : "",
  ]
    .filter(Boolean)
    .join("\n\n");
};
