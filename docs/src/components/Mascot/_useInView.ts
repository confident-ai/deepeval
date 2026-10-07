import { useEffect, useState } from "react";

export const useInView = ({
  rootMargin,
  threshold,
}: { rootMargin?: string; threshold?: number } = {}) => {
  const [node, setNode] = useState<Element | null>(null);
  const [inView, setInView] = useState(false);

  useEffect(() => {
    if (!node || typeof IntersectionObserver === "undefined") return;
    const observer = new IntersectionObserver(
      ([entry]) => setInView(entry.isIntersecting),
      { rootMargin, threshold }
    );
    observer.observe(node);
    return () => observer.disconnect();
  }, [node, rootMargin, threshold]);

  return { ref: setNode, inView };
};
