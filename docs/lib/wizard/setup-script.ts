// Mirrors confident-landing's `functions/wizard/setup-script.ts`, which serves
// https://www.confident-ai.com/setup.sh from the same GitHub releases.
// The only intended difference is the `--from deepeval` entry point below, so
// keep the rest of the script in sync when either copy changes.
export const setupScript = `#!/bin/sh
set -eu

REPO="confident-ai/confident-setup"
BINARY="confident-setup"
RELEASES_URL="https://github.com/\${REPO}/releases"
API_URL="https://api.github.com/repos/\${REPO}/releases/latest"
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

# A leading positional argument pins a release. Options belong to the wizard.
tag=""
if [ "$#" -gt 0 ]; then
  case "$1" in
    -*) ;;
    *) tag="$1"; shift ;;
  esac
fi

if [ -z "$tag" ]; then
  release_json=$(curl -fsSL \\
    -H "Accept: application/vnd.github+json" \\
    -H "X-GitHub-Api-Version: 2022-11-28" \\
    "$API_URL") || fail "could not query the latest stable release"
  tag=$(printf '%s\\n' "$release_json" |
    sed -n 's/.*"tag_name"[[:space:]]*:[[:space:]]*"\\([^"]*\\)".*/\\1/p' |
    sed -n '1p')
  [ -n "$tag" ] || fail "latest release response did not include tag_name"
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
tmpdir=$(mktemp -d "\${TMPDIR:-/tmp}/confident-setup.XXXXXX") ||
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
