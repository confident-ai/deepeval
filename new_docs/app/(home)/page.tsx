import Link from 'next/link';

export default function HomePage() {
  return (
    <main className="flex flex-col items-center justify-center flex-1 gap-6 text-center px-6 py-24">
      <h1 className="text-4xl font-bold tracking-tight">
        The LLM Evaluation Framework
      </h1>
      <p className="max-w-xl text-fd-muted-foreground">
        DeepEval lets you build reliable evaluation pipelines for any LLM
        system. This is a stub homepage &mdash; the real one ships later.
      </p>
      <Link
        href="/docs/getting-started"
        className="rounded-md bg-fd-primary text-fd-primary-foreground px-5 py-2.5 font-medium"
      >
        Get Started
      </Link>
    </main>
  );
}
