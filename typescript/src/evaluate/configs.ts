// Mirrors deepeval/evaluate/configs.py — grouped config objects passed to
// `evaluate()`. Only the fields the TS runner honors today are documented as
// such; the rest are placeholders kept for forward-compatibility with Python.

export interface AsyncConfig {
  runAsync?: boolean;
  maxConcurrent?: number;
  throttleValue?: number;
}

export interface DisplayConfig {
  showIndicator?: boolean;
  printResults?: boolean;
  verboseMode?: boolean;
  /** Truncate (hide) passing cases' detail tables, like Python. */
  truncatePassingCases?: boolean;
  /** When set, also write the report to this directory as a file. */
  fileOutputDir?: string;
  /** File format for `fileOutputDir` (Markdown; MDX is the same content). */
  fileType?: "md" | "mdx";
}

export interface ErrorConfig {
  ignoreErrors?: boolean;
  skipOnMissingParams?: boolean;
}

export interface CacheConfig {
  // TODO: result caching (Python's Cache.get_metric_data). Not wired yet.
  writeCache?: boolean;
  useCache?: boolean;
}

export const DEFAULT_ASYNC_CONFIG: Required<AsyncConfig> = {
  runAsync: true,
  maxConcurrent: 20,
  throttleValue: 0,
};

export const DEFAULT_DISPLAY_CONFIG: Required<DisplayConfig> = {
  showIndicator: true,
  printResults: true,
  verboseMode: false,
  truncatePassingCases: true,
  fileOutputDir: "", // empty = no file output
  fileType: "md",
};

export const DEFAULT_ERROR_CONFIG: Required<ErrorConfig> = {
  ignoreErrors: false,
  skipOnMissingParams: false,
};

export const DEFAULT_CACHE_CONFIG: Required<CacheConfig> = {
  writeCache: false,
  useCache: false,
};
