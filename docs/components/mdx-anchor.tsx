import NextLink from "next/link";
import type { ComponentPropsWithoutRef } from "react";

import { externalRelForOutboundHref } from "@/src/utils/outbound-link-rel";

type MdxAnchorProps = ComponentPropsWithoutRef<"a"> & {
  href?: string;
  prefetch?: boolean;
  replace?: boolean;
};

export const MdxAnchor = ({
  href = "#",
  prefetch,
  replace,
  ...props
}: MdxAnchorProps) => {
  const external = href.match(/^\w+:/) !== null || href.startsWith("//");

  if (!external) {
    return (
      <NextLink
        href={href}
        prefetch={prefetch}
        replace={replace}
        {...props}
      />
    );
  }

  const rel = externalRelForOutboundHref(href);

  return <a href={href} {...props} rel={rel} target="_blank" />;
};
