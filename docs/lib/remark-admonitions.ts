import { visit } from "unist-util-visit";
import { toString as mdastToString } from "mdast-util-to-string";
import type { Root } from "mdast";
import type { ContainerDirective } from "mdast-util-directive";

const ADMONITION_TYPES = new Set([
  "note",
  "info",
  "tip",
  "success",
  "important",
  "warning",
  "caution",
  "danger",
  "error",
  "secondary",
]);

/**
 * Converts Docusaurus-style `:::type[title]` container directives into
 * `<Callout type="..." title="...">` MDX JSX elements. Requires
 * `remark-directive` to run before this plugin.
 */
export function remarkAdmonitions() {
  return (tree: Root) => {
    visit(tree, "containerDirective", (node: ContainerDirective, index, parent) => {
      if (!ADMONITION_TYPES.has(node.name)) return;
      if (!parent || index == null) return;

      // The label (from `:::note[My Title]`) lives as the first child
      // paragraph with `data.directiveLabel` — pluck it out.
      let title: string | undefined;
      const children = [...(node.children ?? [])];
      const labelIdx = children.findIndex(
        (child) =>
          child.type === "paragraph" && (child as { data?: { directiveLabel?: boolean } }).data?.directiveLabel,
      );
      if (labelIdx !== -1) {
        const [label] = children.splice(labelIdx, 1);
        title = mdastToString(label).trim();
      }

      const attributes: Array<{
        type: "mdxJsxAttribute";
        name: string;
        value: string;
      }> = [{ type: "mdxJsxAttribute", name: "type", value: node.name }];
      if (title) {
        attributes.push({ type: "mdxJsxAttribute", name: "title", value: title });
      }

      const replacement = {
        type: "mdxJsxFlowElement" as const,
        name: "Callout",
        attributes,
        children,
      };

      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      parent.children.splice(index, 1, replacement as any);
    });
  };
}

export default remarkAdmonitions;
