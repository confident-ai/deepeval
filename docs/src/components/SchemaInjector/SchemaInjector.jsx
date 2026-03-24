import React from 'react';
import Head from '@docusaurus/Head';

export default function SchemaInjector({ schema }) {
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