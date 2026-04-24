declare module '*.module.css' {
  const classes: { readonly [key: string]: string };
  export default classes;
}

declare module '*.module.scss' {
  const classes: { readonly [key: string]: string };
  export default classes;
}

declare module '*.module.sass' {
  const classes: { readonly [key: string]: string };
  export default classes;
}

declare module '*.mdx' {
  import type { ComponentType } from 'react';
  import type { MDXProps } from 'mdx/types';

  const MDXContent: ComponentType<MDXProps>;
  export default MDXContent;
}