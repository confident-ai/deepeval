"""Download, verify, cache and run the `deepeval-setup` wizard binary.

The wizard lives in `setup-wizard/` and ships as a standalone executable on
GitHub releases tagged `setup-v<version>`. Every run asks GitHub for the newest
of those releases, so a wizard fix reaches users without a DeepEval upgrade, and
falls back to the newest cached copy when GitHub cannot be reached.
"""

import contextlib
import hashlib
import io
import os
import platform
import re
import shutil
import signal
import subprocess
import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from typing import (
    Any,
    Callable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

import requests

BINARY_NAME = "deepeval-setup"
CHECKSUMS_NAME = "SHA256SUMS"
RELEASES_URL = "https://github.com/confident-ai/deepeval/releases/download"
RELEASES_API_URL = "https://api.github.com/repos/confident-ai/deepeval/releases"
# (connect, read) seconds; the read timeout applies per chunk, not overall.
DOWNLOAD_TIMEOUT = (10, 60)
LOOKUP_TIMEOUT = (10, 30)
# Python and TypeScript SDK releases share the list, newest first, so the
# newest wizard release is normally on the first page.
RELEASES_PER_PAGE = 100
LOOKUP_PAGES = 3
_TAG = re.compile(r"^setup-v(\d+)\.(\d+)\.(\d+)$")
_VERSION = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")

_OS_NAMES = {"darwin": "darwin", "linux": "linux"}
_ARCH_NAMES = {
    "x86_64": "x64",
    "amd64": "x64",
    "arm64": "arm64",
    "aarch64": "arm64",
}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class SetupLauncherError(Exception):
    """A failure to report as a plain message, without a traceback."""


class UnsupportedPlatformError(SetupLauncherError):
    pass


def resolve_target(
    system: Optional[str] = None, machine: Optional[str] = None
) -> Tuple[str, str]:
    """Map this machine onto a release asset's `<os>-<arch>`."""
    system = (platform.system() if system is None else system).lower()
    machine = (platform.machine() if machine is None else machine).lower()
    os_name = _OS_NAMES.get(system)
    arch = _ARCH_NAMES.get(machine)
    if os_name is None or arch is None:
        raise UnsupportedPlatformError(
            "deepeval setup supports macOS and Linux on x64 and arm64 for "
            f"now, not {system or 'this system'} on {machine or 'this CPU'}."
        )
    return os_name, arch


def archive_name(os_name: str, arch: str) -> str:
    return f"{BINARY_NAME}-{os_name}-{arch}.tar.gz"


def release_url(name: str, version: str) -> str:
    return f"{RELEASES_URL}/setup-v{version}/{name}"


def cache_root(env: Optional[Mapping[str, str]] = None) -> Path:
    """`${XDG_CACHE_HOME:-~/.cache}/deepeval/setup`."""
    env = os.environ if env is None else env
    base = env.get("XDG_CACHE_HOME") or ""
    # The XDG spec says a relative value is invalid and should be ignored.
    root = Path(base) if os.path.isabs(base) else Path.home() / ".cache"
    return root / "deepeval" / "setup"


def cache_dir(version: str, env: Optional[Mapping[str, str]] = None) -> Path:
    return cache_root(env) / version


def _download(url: str) -> bytes:
    response = requests.get(url, timeout=DOWNLOAD_TIMEOUT)
    response.raise_for_status()
    return response.content


def _fetch_releases(page: int) -> List[Any]:
    response = requests.get(
        RELEASES_API_URL,
        params={"per_page": RELEASES_PER_PAGE, "page": page},
        headers={"Accept": "application/vnd.github+json"},
        timeout=LOOKUP_TIMEOUT,
    )
    response.raise_for_status()
    releases = response.json()
    if not isinstance(releases, list):
        raise SetupLauncherError("GitHub returned an unexpected release list.")
    return releases


def _is_executable(path: Path) -> bool:
    return path.is_file() and os.access(path, os.X_OK)


def latest_version(
    asset: str,
    fetch_releases: Optional[Callable[[int], List[Any]]] = None,
) -> str:
    """The newest published `setup-v*` release that ships `asset`.

    Drafts and prereleases are skipped, so a release can be staged as a
    prerelease before it reaches everyone."""
    fetch_releases = fetch_releases or _fetch_releases
    newest: Optional[Tuple[int, ...]] = None
    for page in range(1, LOOKUP_PAGES + 1):
        releases = fetch_releases(page)
        for release in releases:
            if not isinstance(release, dict):
                continue
            if release.get("draft") or release.get("prerelease"):
                continue
            match = _TAG.match(str(release.get("tag_name") or ""))
            if match is None:
                continue
            assets = {
                item.get("name")
                for item in release.get("assets") or []
                if isinstance(item, dict)
            }
            if asset not in assets or CHECKSUMS_NAME not in assets:
                continue
            version = tuple(int(part) for part in match.groups())
            if newest is None or version > newest:
                newest = version
        if newest is not None or len(releases) < RELEASES_PER_PAGE:
            break
    if newest is None:
        raise SetupLauncherError(
            f"No published setup wizard release ships {asset}."
        )
    return ".".join(str(part) for part in newest)


def newest_cached(env: Optional[Mapping[str, str]] = None) -> Optional[Path]:
    """The newest wizard already in the cache, for when GitHub is out of
    reach."""
    root = cache_root(env)
    if not root.is_dir():
        return None
    cached = []
    for entry in root.iterdir():
        match = _VERSION.match(entry.name)
        binary = entry / BINARY_NAME
        if match and _is_executable(binary):
            cached.append((tuple(int(part) for part in match.groups()), binary))
    return max(cached)[1] if cached else None


def expected_sha256(checksums: str, name: str) -> str:
    """Read `name`'s digest from `sha256sum` output."""
    for line in checksums.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1].lstrip("*") == name:
            digest = parts[0].lower()
            if _SHA256.match(digest):
                return digest
    raise SetupLauncherError(f"{CHECKSUMS_NAME} has no valid entry for {name}.")


def _binary_member(archive: tarfile.TarFile) -> tarfile.TarInfo:
    """The single regular `deepeval-setup` file, checked by hand because
    Python 3.9 has no extraction filters."""
    found: Optional[tarfile.TarInfo] = None
    for member in archive.getmembers():
        path = PurePosixPath(member.name)
        if path.is_absolute() or ".." in path.parts:
            raise SetupLauncherError(
                f"The setup archive contains an unsafe path: {member.name!r}."
            )
        if path.parts != (BINARY_NAME,):
            continue
        if not member.isreg():
            raise SetupLauncherError(
                f"{BINARY_NAME} in the setup archive is not a regular file."
            )
        if found is not None:
            raise SetupLauncherError(
                f"The setup archive contains {BINARY_NAME} more than once."
            )
        found = member
    if found is None:
        raise SetupLauncherError(
            f"The setup archive does not contain {BINARY_NAME}."
        )
    return found


@contextlib.contextmanager
def _atomic_target(target: Path) -> Iterator[io.BufferedWriter]:
    """Write beside `target`, then rename over it, so an interrupted write
    never leaves a partial executable where the cache expects a whole one."""
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{BINARY_NAME}.", dir=str(target.parent)
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            yield handle
        os.chmod(temporary, 0o755)
        os.replace(temporary, target)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary)
        raise


def install_binary(
    archive: bytes, checksums: str, name: str, target: Path
) -> Path:
    """Verify `archive` against `checksums`, then extract the binary to
    `target`. Nothing is written unless the digest matches."""
    actual = hashlib.sha256(archive).hexdigest()
    if actual != expected_sha256(checksums, name):
        raise SetupLauncherError(
            f"Checksum mismatch for {name}; refusing to run it. "
            "Try again, and report it if it persists."
        )
    try:
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tar:
            source = tar.extractfile(_binary_member(tar))
            if source is None:
                raise SetupLauncherError(f"Could not read {BINARY_NAME}.")
            with source, _atomic_target(target) as handle:
                shutil.copyfileobj(source, handle)
    except (tarfile.TarError, OSError) as error:
        # OSError covers both a corrupt gzip stream and an unwritable cache.
        raise SetupLauncherError(
            f"Could not install the setup wizard into {target.parent}: {error}"
        ) from error
    return target


def ensure_binary(
    *,
    download: Optional[Callable[[str], bytes]] = None,
    fetch_releases: Optional[Callable[[int], List[Any]]] = None,
    system: Optional[str] = None,
    machine: Optional[str] = None,
    env: Optional[Mapping[str, str]] = None,
    notify: Callable[[str], None] = lambda _message: None,
) -> Path:
    """Return the newest wizard binary, downloading it first if needed."""
    # Resolved per call, so patching the helpers reaches every caller.
    download = download or _download
    fetch_releases = fetch_releases or _fetch_releases
    os_name, arch = resolve_target(system, machine)
    name = archive_name(os_name, arch)

    try:
        version = latest_version(name, fetch_releases)
    except (requests.RequestException, ValueError, SetupLauncherError) as error:
        cached = newest_cached(env)
        if cached is None:
            raise SetupLauncherError(
                f"Could not find the setup wizard on GitHub: {error}"
            ) from error
        notify(
            f"Could not check GitHub for a newer setup wizard ({error}); "
            f"using the cached {cached.parent.name}."
        )
        return cached

    target = cache_dir(version, env) / BINARY_NAME
    if _is_executable(target):
        return target

    notify(
        f"Downloading the DeepEval setup wizard {version} "
        f"({os_name}-{arch})..."
    )
    try:
        archive = download(release_url(name, version))
        checksums = download(release_url(CHECKSUMS_NAME, version))
    except requests.RequestException as error:
        raise SetupLauncherError(
            f"Could not download the setup wizard: {error}"
        ) from error
    return install_binary(
        archive, checksums.decode("utf-8", errors="replace"), name, target
    )


def _exit_code(returncode: int) -> int:
    # A child killed by a signal reports -N; shells report that as 128 + N.
    return 128 - returncode if returncode < 0 else returncode


def run_binary(
    binary: Path,
    args: Sequence[str],
    runner: Optional[
        Callable[..., "subprocess.CompletedProcess[bytes]"]
    ] = None,
) -> int:
    """Run the wizard on this terminal and return its exit code.

    Ctrl+C reaches the whole foreground process group, and the wizard handles
    it itself, so the parent must not die (or kill the child, as
    `subprocess.run` does on KeyboardInterrupt) while it is running. A Python
    handler, unlike SIG_IGN, is reset to the default in the child on exec.
    """
    runner = runner or subprocess.run
    installed = False
    previous = None
    with contextlib.suppress(ValueError):  # Not on the main thread.
        previous = signal.signal(signal.SIGINT, lambda *_: None)
        installed = True
    try:
        completed = runner([str(binary), *args])
    except OSError as error:
        raise SetupLauncherError(
            f"Could not start the setup wizard at {binary}: {error}"
        ) from error
    finally:
        if installed:
            signal.signal(
                signal.SIGINT,
                signal.SIG_DFL if previous is None else previous,
            )
    return _exit_code(completed.returncode)
