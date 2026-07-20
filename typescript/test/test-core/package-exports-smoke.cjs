const fs = require("fs");
const os = require("os");
const path = require("path");
const { execFileSync } = require("child_process");

const packageRoot = path.resolve(__dirname, "../..");
const packageJson = require(path.join(packageRoot, "package.json"));

const packageSubpaths = Object.keys(packageJson.exports)
  .filter((exportKey) => exportKey !== ".")
  .map((exportKey) => ({
    exportKey,
    specifier: `deepeval/${exportKey.slice(2)}`,
  }));

const loadablePackageSubpaths = [
  "deepeval/annotation",
  "deepeval/metrics",
  "deepeval/models",
  "deepeval/test-case",
  "deepeval/evaluate",
  "deepeval/testCase",
];

const privateSubpaths = ["deepeval/annotation/utils"];

const optionalPeerSubpaths = [
  {
    specifier: "deepeval/integrations/langchain",
    missingPeerSpecifier: "@langchain/core",
    missingPeerPattern: /@langchain\/core/,
  },
];

const optionalPeerSpecifiers = new Set(
  optionalPeerSubpaths.map(({ specifier }) => specifier),
);

const consumerLoadableSubpaths = packageSubpaths
  .map(({ specifier }) => specifier)
  .filter((specifier) => !optionalPeerSpecifiers.has(specifier));

function assert(condition, message) {
  if (!condition) {
    throw new Error(message);
  }
}

function formatSubpaths(subpaths) {
  return subpaths.length > 0 ? subpaths.join(", ") : "<none>";
}

function assertExportFilesExist(exportKey) {
  const exportEntry = packageJson.exports[exportKey];
  assert(exportEntry, `Missing package export ${exportKey}`);

  for (const field of ["import", "require", "types"]) {
    const relativePath = exportEntry[field];
    assert(
      fs.existsSync(path.resolve(packageRoot, relativePath)),
      `Missing ${field} file for ${exportKey}: ${relativePath}`,
    );
  }
}

function run(command, args, options = {}) {
  execFileSync(command, args, {
    stdio: "pipe",
    encoding: "utf8",
    ...options,
  });
}

function loadSpecifier(specifier) {
  require(specifier);
  return import(specifier);
}

async function assertOptionalPeerImport(specifier, missingPeerPattern) {
  try {
    await loadSpecifier(specifier);
  } catch (error) {
    assert(
      error.code === "MODULE_NOT_FOUND" &&
        missingPeerPattern.test(error.message),
      `${specifier} failed for an unexpected reason: ${error.message}`,
    );
    return;
  }
}

function createConsumerScript() {
  return `
const assert = (condition, message) => {
  if (!condition) throw new Error(message);
};

const loadableSubpaths = ${JSON.stringify(consumerLoadableSubpaths, null, 2)};
const optionalPeerSubpaths = ${JSON.stringify(
    optionalPeerSubpaths.map(
      ({ specifier, missingPeerSpecifier, missingPeerPattern }) => ({
        specifier,
        missingPeerSpecifier,
        missingPeerPattern: missingPeerPattern.source,
      }),
    ),
    null,
    2,
  )};

(async () => {
  for (const specifier of loadableSubpaths) {
    require.resolve(specifier);
    require(specifier);
    await import(specifier);
  }

  for (const { specifier, missingPeerSpecifier, missingPeerPattern } of optionalPeerSubpaths) {
    require.resolve(specifier);

    let peerResolveError;
    try {
      require.resolve(missingPeerSpecifier);
    } catch (error) {
      peerResolveError = error;
    }

    assert(
      peerResolveError && peerResolveError.code === "MODULE_NOT_FOUND",
      missingPeerSpecifier + " should not be installed in the packed consumer smoke test",
    );

    let commonJsError;
    try {
      require(specifier);
    } catch (error) {
      commonJsError = error;
    }

    const expectedMissingPeer = new RegExp(missingPeerPattern);
    assert(
      commonJsError &&
        commonJsError.code === "MODULE_NOT_FOUND" &&
        expectedMissingPeer.test(commonJsError.message),
      specifier + " should fail with missing optional peer " + missingPeerSpecifier,
    );

    let esmError;
    try {
      await import(specifier);
    } catch (error) {
      esmError = error;
    }

    assert(
      esmError &&
        esmError.code === "MODULE_NOT_FOUND" &&
        expectedMissingPeer.test(esmError.message),
      specifier + " ESM import should fail with missing optional peer " + missingPeerSpecifier,
    );
  }
})().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
`;
}

function assertPackedArtifactConsumerInstall() {
  const packDir = fs.mkdtempSync(path.join(os.tmpdir(), "deepeval-pack-"));
  const consumerDir = fs.mkdtempSync(
    path.join(os.tmpdir(), "deepeval-consumer-"),
  );

  try {
    const tarball = execFileSync(
      "npm",
      ["pack", "--pack-destination", packDir, "--silent"],
      { cwd: packageRoot, encoding: "utf8" },
    ).trim();
    const tarballPath = path.join(packDir, tarball);

    fs.writeFileSync(
      path.join(consumerDir, "package.json"),
      JSON.stringify({ private: true }, null, 2),
    );

    run(
      "npm",
      [
        "install",
        "--ignore-scripts",
        "--omit=dev",
        "--no-audit",
        "--no-fund",
        tarballPath,
      ],
      { cwd: consumerDir },
    );

    run("node", ["-e", createConsumerScript()], { cwd: consumerDir });
  } finally {
    fs.rmSync(packDir, { recursive: true, force: true });
    fs.rmSync(consumerDir, { recursive: true, force: true });
  }
}

async function main() {
  const typesVersionSubpaths = Object.keys(packageJson.typesVersions["*"])
    .filter((subpath) => subpath !== "*")
    .map((subpath) => `./${subpath}`);

  const exportKeys = packageSubpaths.map(({ exportKey }) => exportKey);
  const missingTypesVersions = exportKeys.filter(
    (exportKey) => !typesVersionSubpaths.includes(exportKey),
  );
  const extraTypesVersions = typesVersionSubpaths.filter(
    (subpath) => !exportKeys.includes(subpath),
  );

  assert(
    missingTypesVersions.length === 0 && extraTypesVersions.length === 0,
    [
      "Package exports and typesVersions drift detected.",
      `Missing typesVersions entries for exports: ${formatSubpaths(missingTypesVersions)}`,
      `Extra typesVersions entries without exports: ${formatSubpaths(extraTypesVersions)}`,
    ].join("\n"),
  );

  for (const { exportKey, specifier } of packageSubpaths) {
    assertExportFilesExist(exportKey);
    require.resolve(specifier, { paths: [packageRoot] });
    assert(
      typesVersionSubpaths.includes(exportKey),
      `Missing typesVersions entry for ${exportKey}`,
    );
  }

  for (const specifier of loadablePackageSubpaths) {
    await loadSpecifier(specifier);
  }

  for (const { specifier, missingPeerPattern } of optionalPeerSubpaths) {
    await assertOptionalPeerImport(specifier, missingPeerPattern);
  }

  for (const specifier of privateSubpaths) {
    try {
      require.resolve(specifier, { paths: [packageRoot] });
      throw new Error(`Expected ${specifier} to be private`);
    } catch (error) {
      if (error.code !== "ERR_PACKAGE_PATH_NOT_EXPORTED") {
        throw error;
      }
    }
  }

  assertPackedArtifactConsumerInstall();
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
