"use client";

import type { ReactNode } from "react";

type NotebookSectionLayoutProps = {
  heading: ReactNode;
  children: ReactNode;
};

const NotebookSectionLayout: React.FC<NotebookSectionLayoutProps> = ({
  heading,
  children,
}) => {
  return (
    <section className="w-full">
      <h2 className="text-xl font-semibold text-fd-foreground">{heading}</h2>
      <div className="mt-3">{children}</div>
    </section>
  );
};


export default NotebookSectionLayout;
