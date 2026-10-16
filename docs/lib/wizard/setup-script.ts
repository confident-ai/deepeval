// Installs and runs DeepEval's own setup wizard, `deepeval-setup`, from this
// repository's `setup-v*` releases: the same ones `deepeval setup` and
// `npx deepeval setup` download, so keep the archive names and the choice of
// release in step with `deepeval/cli/setup/launcher.py` and
// `typescript/src/cli/setup/launcher.ts`. confident-landing's script installs
// Confident AI's tracing wizard instead and is no longer a copy of this one.
export const setupScript = `#!/bin/sh
set -eu

REPO="confident-ai/deepeval"
BINARY="deepeval-setup"
RELEASES_URL="https://github.com/\${REPO}/releases"
API_URL="https://api.github.com/repos/\${REPO}/releases?per_page=100"
SOURCE="deepeval"

red() {
  printf '\\033[0;31m%s\\033[0m\\n' "$*" >&2
}

fail() {
  red "Error: $*"
  exit 1
}

# Status belongs on stderr so the wizard keeps stdout to itself, and stays plain
# when nothing can render color.
note() {
  if [ -t 2 ] && [ -z "\${NO_COLOR:-}" ]; then
    printf '\\033[0;36m%s\\033[0m\\n' "$*" >&2
  else
    printf '%s\\n' "$*" >&2
  fi
}

command -v curl >/dev/null 2>&1 || fail "curl is required"
command -v tar >/dev/null 2>&1 || fail "tar is required"

# A leading positional argument pins a release, as setup-v0.1.0 or 0.1.0.
# Options belong to the wizard.
tag=""
if [ "$#" -gt 0 ]; then
  case "$1" in
    -*) ;;
    *) tag="$1"; shift ;;
  esac
fi
case "$tag" in
  ""|setup-v*) ;;
  *) tag="setup-v\${tag}" ;;
esac

# This repository's "latest" release is the Python SDK, so take the highest
# published setup-v* release instead, skipping prereleases as the launchers do.
# Splitting on commas and braces puts one field per line, whether GitHub sends
# the list compact or indented.
if [ -z "$tag" ]; then
  releases_json=$(curl -fsSL \\
    -H "Accept: application/vnd.github+json" \\
    -H "X-GitHub-Api-Version: 2022-11-28" \\
    "$API_URL") || fail "could not list releases"
  version=$(printf '%s\\n' "$releases_json" | tr ',{}' '\\n\\n\\n' | awk '
    /"tag_name"[[:space:]]*:/ {
      tag = $0
      sub(/.*"tag_name"[[:space:]]*:[[:space:]]*"/, "", tag)
      sub(/".*/, "", tag)
    }
    /"prerelease"[[:space:]]*:[[:space:]]*false/ &&
      tag ~ /^setup-v[0-9]+[.][0-9]+[.][0-9]+$/ {
      print substr(tag, 8)
    }
  ' | sort -t. -k1,1n -k2,2n -k3,3n | tail -n 1)
  [ -n "$version" ] || fail "no published setup wizard release was found"
  tag="setup-v\${version}"
fi

case "$tag" in
  *[!A-Za-z0-9._-]*) fail "invalid release tag: $tag" ;;
esac

case "$(uname -s)" in
  Darwin) os="darwin" ;;
  Linux) os="linux" ;;
  *) fail "unsupported operating system: $(uname -s)" ;;
esac

case "$(uname -m)" in
  x86_64|amd64) arch="x64" ;;
  arm64|aarch64) arch="arm64" ;;
  *) fail "unsupported architecture: $(uname -m)" ;;
esac

archive="\${BINARY}-\${os}-\${arch}.tar.gz"
download_url="\${RELEASES_URL}/download/\${tag}"
tmpdir=$(mktemp -d "\${TMPDIR:-/tmp}/deepeval-setup.XXXXXX") ||
  fail "could not create a temporary directory"

cleanup() {
  rm -rf "$tmpdir"
}
trap cleanup 0
trap 'exit 1' HUP INT TERM

# The archive is tens of megabytes, and downloading it in silence is
# indistinguishable from a hang, so curl draws its own bar when a terminal is
# there to receive it.
if [ -t 2 ]; then
  progress="--progress-bar"
else
  progress="--silent"
fi

note "Downloading \${BINARY} \${tag} for \${os}-\${arch}…"
curl -fL --retry 3 --retry-delay 1 "\$progress" \\
  "\${download_url}/\${archive}" \\
  -o "\${tmpdir}/\${archive}" ||
  fail "could not download \${archive} for release \${tag}"
curl -fsSL --retry 3 --retry-delay 1 \\
  "\${download_url}/SHA256SUMS" \\
  -o "\${tmpdir}/SHA256SUMS" ||
  fail "could not download SHA256SUMS for release \${tag}"

expected=$(awk -v file="$archive" '
  $2 == file || $2 == ("*" file) { print $1; exit }
' "\${tmpdir}/SHA256SUMS")
[ -n "$expected" ] ||
  fail "SHA256SUMS does not contain a checksum for \${archive}"

if command -v shasum >/dev/null 2>&1; then
  actual=$(shasum -a 256 "\${tmpdir}/\${archive}" | awk '{ print $1 }')
elif command -v sha256sum >/dev/null 2>&1; then
  actual=$(sha256sum "\${tmpdir}/\${archive}" | awk '{ print $1 }')
else
  fail "shasum or sha256sum is required to verify the download"
fi

[ "$actual" = "$expected" ] ||
  fail "checksum verification failed for \${archive}"

mkdir -p "\${tmpdir}/extract"
tar -xzf "\${tmpdir}/\${archive}" -C "\${tmpdir}/extract" ||
  fail "could not extract \${archive}"

extracted="\${tmpdir}/extract/\${BINARY}"
[ -f "$extracted" ] ||
  fail "the archive did not contain \${BINARY}"

bin_dir="\${XDG_BIN_HOME:-\${HOME:-}/.local/bin}"
[ -n "\${HOME:-}" ] || [ -n "\${XDG_BIN_HOME:-}" ] ||
  fail "HOME or XDG_BIN_HOME must be set"
mkdir -p "$bin_dir" || fail "could not create $bin_dir"

installed="\${bin_dir}/\${BINARY}"
note "Installing \${BINARY} to \${bin_dir}…"
if command -v install >/dev/null 2>&1; then
  install -m 755 "$extracted" "$installed" ||
    fail "could not install \${BINARY} to \${bin_dir}"
else
  cp "$extracted" "$installed" ||
    fail "could not install \${BINARY} to \${bin_dir}"
  chmod 755 "$installed"
fi

cleanup
trap - 0

# curl | sh consumes stdin, so the wizard needs the terminal instead. Only the
# command being exec'd may take it: this shell is still reading the rest of the
# script from stdin, and redirecting the shell itself would make it read the
# remainder from the terminal and wait for the user to type it.
has_tty=0
if (exec </dev/tty) 2>/dev/null; then
  has_tty=1
fi

# BSD script gives interactive macOS sessions a real pseudo-terminal.
if [ "$os" = "darwin" ] && [ "$has_tty" -eq 1 ] &&
  command -v script >/dev/null 2>&1; then
  exec script -q /dev/null "$installed" --from "$SOURCE" "$@" </dev/tty
fi

if [ "$has_tty" -eq 1 ]; then
  exec "$installed" --from "$SOURCE" "$@" </dev/tty
fi

exec "$installed" --from "$SOURCE" "$@"
`;
