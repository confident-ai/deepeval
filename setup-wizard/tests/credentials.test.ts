import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, describe, expect, it, vi } from "vitest";

import {
  apiUrlForKey,
  checkExistingKey,
  connectConfidentAi,
  findExistingKey,
  KEY_CHECK_PATH,
  type ConnectDependencies,
  type ExistingKey,
} from "../src/credentials.js";

const directories: string[] = [];

const projectWithEnvLocal = async (content?: string): Promise<string> => {
  const directory = await mkdtemp(join(tmpdir(), "deepeval-setup-creds-"));
  directories.push(directory);
  if (content !== undefined) {
    await writeFile(join(directory, ".env.local"), content, { mode: 0o600 });
  }
  return directory;
};

afterEach(async () => {
  await Promise.all(
    directories
      .splice(0)
      .map((directory) => rm(directory, { recursive: true, force: true })),
  );
});

const API_URL = "https://api.confident-ai.com";
const KEY = "confident_us_proj_c2VjcmV0";

const respond = (status: number) =>
  vi
    .fn<typeof globalThis.fetch>()
    .mockImplementation(() => Promise.resolve(new Response("{}", { status })));

/**
 * Wires the real check to a mocked fetch, so each test sees every request the
 * flow would send, while sign-in stands in for the pairing calls.
 */
const dependencies = (
  fetch: typeof globalThis.fetch,
  key: ExistingKey | undefined,
  reuse = true,
) => {
  const signIn = vi
    .fn<ConnectDependencies["signIn"]>()
    .mockResolvedValue({ apiKey: "new-key", projectId: "new-project" });
  const chooseReuse = vi
    .fn<ConnectDependencies["chooseReuse"]>()
    .mockResolvedValue(reuse);
  return {
    signIn,
    chooseReuse,
    deps: {
      findKey: () => Promise.resolve(key),
      chooseReuse,
      checkKey: (apiUrl: string, apiKey: string, projectId?: string) =>
        checkExistingKey(apiUrl, apiKey, projectId, fetch),
      signIn,
    } satisfies ConnectDependencies,
  };
};

describe("findExistingKey", () => {
  it("prefers the environment, as DeepEval does", async () => {
    const directory = await projectWithEnvLocal(
      'CONFIDENT_API_KEY="from-file"\n',
    );
    await expect(
      findExistingKey(directory, { CONFIDENT_API_KEY: " from-env " }),
    ).resolves.toEqual({ apiKey: "from-env", source: "environment" });
  });

  it("falls back to .env.local", async () => {
    const directory = await projectWithEnvLocal(
      'OTHER=1\nexport CONFIDENT_API_KEY="from-file"\n',
    );
    await expect(
      findExistingKey(directory, { CONFIDENT_API_KEY: "  " }),
    ).resolves.toEqual({ apiKey: "from-file", source: ".env.local" });
  });

  it("finds nothing when neither has a key", async () => {
    await expect(
      findExistingKey(await projectWithEnvLocal(), {}),
    ).resolves.toBeUndefined();
    await expect(
      findExistingKey(await projectWithEnvLocal("CONFIDENT_API_KEY=\n"), {}),
    ).resolves.toBeUndefined();
  });
});

describe("checkExistingKey", () => {
  it("sends the key in the SDK header and no project unless asked", async () => {
    const fetch = respond(200);
    await expect(
      checkExistingKey(`${API_URL}/`, KEY, undefined, fetch),
    ).resolves.toEqual({ status: "valid" });
    const [url, init] = fetch.mock.calls[0]!;
    expect(url).toBe(`${API_URL}${KEY_CHECK_PATH}`);
    expect(init?.headers).toEqual({ CONFIDENT_API_KEY: KEY });
  });

  it("treats a plan or feature gate as a valid key", async () => {
    await expect(
      checkExistingKey(API_URL, KEY, undefined, respond(403)),
    ).resolves.toEqual({ status: "valid" });
  });

  it("reads 401 as a rejected key", async () => {
    await expect(
      checkExistingKey(API_URL, KEY, undefined, respond(401)),
    ).resolves.toEqual({ status: "rejected" });
  });

  it("reads 400 with a project id as the wrong project", async () => {
    const fetch = respond(400);
    await expect(
      checkExistingKey(API_URL, KEY, "other-project", fetch),
    ).resolves.toEqual({ status: "wrong-project" });
    expect(fetch.mock.calls[0]![1]?.headers).toEqual({
      CONFIDENT_API_KEY: KEY,
      "confident-project-id": "other-project",
    });
  });

  it("never echoes a network error", async () => {
    const fetch = vi
      .fn<typeof globalThis.fetch>()
      .mockRejectedValue(new Error(`boom ${KEY}`));
    const check = await checkExistingKey(API_URL, KEY, undefined, fetch);
    expect(check.status).toBe("unverified");
    expect(JSON.stringify(check)).not.toContain(KEY);
  });

  it("routes an EU key to the EU API only from the default URL", () => {
    expect(apiUrlForKey(API_URL, "confident_eu_proj_abc")).toBe(
      "https://eu.api.confident-ai.com",
    );
    expect(apiUrlForKey(API_URL, KEY)).toBe(API_URL);
    expect(apiUrlForKey("https://api.example", "confident_eu_proj_abc")).toBe(
      "https://api.example",
    );
  });
});

describe("connectConfidentAi", () => {
  const request = { projectDir: "/project", apiUrl: API_URL };

  it("reuses an accepted key without any pairing call", async () => {
    const fetch = respond(200);
    const { deps, signIn } = dependencies(fetch, {
      apiKey: KEY,
      source: ".env.local",
    });
    await expect(connectConfidentAi(request, deps)).resolves.toEqual({
      apiKey: KEY,
      apiUrl: API_URL,
      reused: true,
    });
    expect(signIn).not.toHaveBeenCalled();
    expect(fetch).toHaveBeenCalledTimes(1);
    expect(
      fetch.mock.calls.every(([url]) => !String(url).includes("/cli/")),
    ).toBe(true);
  });

  it("signs in after Confident AI rejects the key", async () => {
    const existing: ExistingKey = { apiKey: KEY, source: "environment" };
    const { deps, signIn } = dependencies(respond(401), existing);
    await expect(connectConfidentAi(request, deps)).resolves.toEqual({
      apiKey: "new-key",
      projectId: "new-project",
      apiUrl: API_URL,
      reused: false,
    });
    expect(signIn).toHaveBeenCalledWith(existing);
  });

  it("signs in when --proj-id names another project", async () => {
    const fetch = respond(400);
    const { deps, signIn } = dependencies(fetch, {
      apiKey: KEY,
      source: ".env.local",
    });
    const credentials = await connectConfidentAi(
      { ...request, projId: "other-project" },
      deps,
    );
    expect(credentials.reused).toBe(false);
    expect(signIn).toHaveBeenCalledOnce();
    expect(fetch.mock.calls[0]![1]?.headers).toMatchObject({
      "confident-project-id": "other-project",
    });
  });

  it("keeps --proj-id as the project when the key belongs to it", async () => {
    const { deps, signIn } = dependencies(respond(200), {
      apiKey: KEY,
      source: ".env.local",
    });
    await expect(
      connectConfidentAi({ ...request, projId: "project" }, deps),
    ).resolves.toMatchObject({ reused: true, projectId: "project" });
    expect(signIn).not.toHaveBeenCalled();
  });

  it("signs in without checking when the user asks for a new key", async () => {
    const fetch = respond(200);
    const { deps, signIn } = dependencies(
      fetch,
      { apiKey: KEY, source: ".env.local" },
      false,
    );
    await expect(connectConfidentAi(request, deps)).resolves.toMatchObject({
      reused: false,
    });
    expect(fetch).not.toHaveBeenCalled();
    expect(signIn).toHaveBeenCalledOnce();
  });

  it("asks nothing when the project has no key", async () => {
    const { deps, signIn, chooseReuse } = dependencies(respond(200), undefined);
    await connectConfidentAi(request, deps);
    expect(chooseReuse).not.toHaveBeenCalled();
    expect(signIn).toHaveBeenCalledWith(undefined);
  });

  it("does not offer a key it cannot match to --org-id", async () => {
    const { deps, signIn, chooseReuse } = dependencies(respond(200), {
      apiKey: KEY,
      source: ".env.local",
    });
    await connectConfidentAi({ ...request, orgId: "org" }, deps);
    expect(chooseReuse).not.toHaveBeenCalled();
    expect(signIn).toHaveBeenCalledWith(undefined);
  });

  it("signs in when Confident AI cannot be reached", async () => {
    const fetch = vi
      .fn<typeof globalThis.fetch>()
      .mockRejectedValue(new Error("fetch failed"));
    const { deps, signIn } = dependencies(fetch, {
      apiKey: KEY,
      source: ".env.local",
    });
    await expect(connectConfidentAi(request, deps)).resolves.toMatchObject({
      reused: false,
    });
    expect(signIn).toHaveBeenCalledOnce();
  });

  it("checks and keeps an EU key against the EU API", async () => {
    const fetch = respond(200);
    const { deps } = dependencies(fetch, {
      apiKey: "confident_eu_proj_abc",
      source: ".env.local",
    });
    await expect(connectConfidentAi(request, deps)).resolves.toMatchObject({
      apiUrl: "https://eu.api.confident-ai.com",
      reused: true,
    });
    expect(fetch.mock.calls[0]![0]).toBe(
      `https://eu.api.confident-ai.com${KEY_CHECK_PATH}`,
    );
  });
});
