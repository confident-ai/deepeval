import React from 'react';
import Head from '@docusaurus/Head';

interface SchemaInjectorProps {
  schema: Record<string, any> | null | undefined;
}

export default function SchemaInjector({ schema }: SchemaInjectorProps) {
  if (!schema || Object.keys(schema).length === 0) {
    return null;
  }

  return (
    <Head>
      <script type="application/ld+json">
        {JSON.stringify(schema)}
      </script>
    </Head>
  );
}