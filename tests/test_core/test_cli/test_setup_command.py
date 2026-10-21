import hashlib
import io
import os
import stat
import subprocess
import tarfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest
import requests
from typer.testing import CliRunner

from deepeval.cli.main import app
from deepeval.cli.setup import launcher

BINARY = b"#!/bin/sh\necho wizard\n"
ARCHIVE_NAME = "deepeval-setup-linux-arm64.tar.gz"
VERSION = "0.2.0"


def _tar_gz(members: Dict[str, Optional[bytes]]) -> bytes:
    """Build an archive in memory; a `None` body makes a symlink."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for name, body in members.items():
            info = tarfile.TarInfo(name)
            if body is None:
                info.type = tarfile.SYMTYPE
                info.linkname = "/etc/passwd"
                tar.addfile(info)
            else:
                info.size = len(body)
                info.mode = 0o644
                tar.addfile(info, io.BytesIO(body))
    return buffer.getvalue()


def _checksums(archive: bytes, name: str = ARCHIVE_NAME) -> bytes:
    digest = hashlib.sha256(archive).hexdigest()
    return f"{'0' * 64}  other.tar.gz\n{digest}  {name}\n".encode()


def _release(
    tag: str,
    assets: Optional[List[str]] = None,
    draft: bool = False,
    prerelease: bool = False,
) -> Dict[str, Any]:
    names = [ARCHIVE_NAME, "SHA256SUMS"] if assets is None else assets
    return {
        "tag_name": tag,
        "draft": draft,
        "prerelease": prerelease,
        "assets": [{"name": name} for name in names],
    }


class FakeGitHub:
    """Stands in for the release list and its downloads, so no test touches
    the network."""

    def __init__(
        self,
        archive: bytes,
        checksums: bytes,
        pages: Optional[List[List[Dict[str, Any]]]] = None,
    ) -> None:
        self.pages = (
            [[_release("python-v4.3.0"), _release(f"setup-v{VERSION}")]]
            if pages is None
            else pages
        )
        self.files = {
            launcher.release_url(ARCHIVE_NAME, VERSION): archive,
            launcher.release_url(launcher.CHECKSUMS_NAME, VERSION): checksums,
        }
        self.pages_requested: List[int] = []
        self.requested: List[str] = []

    def releases(self, page: int) -> List[Dict[str, Any]]:
        self.pages_requested.append(page)
        return self.pages[page - 1] if page <= len(self.pages) else []

    def download(self, url: str) -> bytes:
        self.requested.append(url)
        return self.files[url]


class FakeRunner:
    def __init__(self, returncode: int = 0) -> None:
        self.returncode = returncode
        self.calls: List[List[str]] = []

    def __call__(self, argv: List[str]) -> "subprocess.CompletedProcess[bytes]":
        self.calls.append(argv)
        return subprocess.CompletedProcess(argv, self.returncode)


@pytest.fixture(autouse=True)
def _no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("setup tests must not reach the network")

    monkeypatch.setattr("deepeval.cli.setup.launcher.requests.get", refuse)


@pytest.fixture
def cache_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "xdg-cache"
    monkeypatch.setenv("XDG_CACHE_HOME", str(home))
    return home


@pytest.fixture
def linux_arm64(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "deepeval.cli.setup.launcher.platform.system", lambda: "Linux"
    )
    monkeypatch.setattr(
        "deepeval.cli.setup.launcher.platform.machine", lambda: "aarch64"
    )


def _install(
    monkeypatch: pytest.MonkeyPatch, github: FakeGitHub, runner: FakeRunner
) -> None:
    monkeypatch.setattr(
        "deepeval.cli.setup.launcher._fetch_releases", github.releases
    )
    monkeypatch.setattr(
        "deepeval.cli.setup.launcher._download", github.download
    )
    monkeypatch.setattr("deepeval.cli.setup.launcher.subprocess.run", runner)


def _binary_path(cache_home: Path, version: str = VERSION) -> Path:
    return cache_home / "deepeval" / "setup" / version / "deepeval-setup"


def _cache(cache_home: Path, version: str) -> Path:
    binary = _binary_path(cache_home, version)
    binary.parent.mkdir(parents=True)
    binary.write_bytes(BINARY)
    binary.chmod(0o755)
    return binary


def test_downloads_verifies_caches_and_runs_the_latest_wizard(
    monkeypatch: pytest.MonkeyPatch, cache_home: Path, linux_arm64: None
) -> None:
    archive = _tar_gz({"deepeval-setup": BINARY})
    github = FakeGitHub(archive, _checksums(archive))
    runner = FakeRunner(returncode=7)
    _install(monkeypatch, github, runner)

    result = CliRunner().invoke(
        app,
        [
            "setup",
            "--project-dir",
            "app",
            "--proj-id=project",
            "--from",
            "deepeval",
            "-h",
        ],
    )

    assert result.exit_code == 7, result.output
    binary = _binary_path(cache_home)
    assert binary.read_bytes() == BINARY
    assert stat.S_IMODE(binary.stat().st_mode) == 0o755
    assert github.requested == [
        "https://github.com/confident-ai/deepeval/releases/download/"
        f"setup-v{VERSION}/{ARCHIVE_NAME}",
        "https://github.com/confident-ai/deepeval/releases/download/"
        f"setup-v{VERSION}/SHA256SUMS",
    ]
    assert runner.calls == [
        [
            str(binary),
            "--project-dir",
            "app",
            "--proj-id=project",
            "--from",
            "deepeval",
            "-h",
        ]
    ]
    # Only the binary lands in the cache: no temporary file is left behind.
    assert os.listdir(binary.parent) == ["deepeval-setup"]


def test_picks_the_newest_published_release_that_ships_this_platform() -> None:
    pages = [
        [
            _release("typescript-v0.9.21"),
            _release("setup-v0.4.0", draft=True),
            _release("setup-v0.3.5", prerelease=True),
            _release("setup-v0.3.0", assets=["SHA256SUMS"]),
            _release("setup-v0.10.0"),
            _release("setup-v0.9.0"),
            _release("setup-latest"),
            "not a release",
        ]
    ]
    github = FakeGitHub(b"", b"", pages)

    assert launcher.latest_version(ARCHIVE_NAME, github.releases) == "0.10.0"
    assert github.pages_requested == [1]


def test_looks_past_a_full_page_without_a_wizard_release() -> None:
    full_page = [
        _release(f"python-v4.{index}.0")
        for index in range(launcher.RELEASES_PER_PAGE)
    ]
    github = FakeGitHub(b"", b"", [full_page, [_release("setup-v0.1.0")]])

    assert launcher.latest_version(ARCHIVE_NAME, github.releases) == "0.1.0"
    assert github.pages_requested == [1, 2]


def test_a_cached_latest_wizard_skips_the_download(
    monkeypatch: pytest.MonkeyPatch, cache_home: Path, linux_arm64: None
) -> None:
    binary = _cache(cache_home, VERSION)
    github = FakeGitHub(b"", b"")
    runner = FakeRunner()
    _install(monkeypatch, github, runner)

    result = CliRunner().invoke(app, ["setup", "--org-id", "org"])

    assert result.exit_code == 0, result.output
    assert github.requested == []
    assert runner.calls == [[str(binary), "--org-id", "org"]]


def test_a_newer_release_replaces_an_older_cached_wizard(
    monkeypatch: pytest.MonkeyPatch, cache_home: Path, linux_arm64: None
) -> None:
    _cache(cache_home, "0.1.0")
    archive = _tar_gz({"deepeval-setup": BINARY})
    github = FakeGitHub(archive, _checksums(archive))
    runner = FakeRunner()
    _install(monkeypatch, github, runner)

    assert CliRunner().invoke(app, ["setup"]).exit_code == 0
    assert len(github.requested) == 2
    assert runner.calls == [[str(_binary_path(cache_home))]]


def test_falls_back_to_the_newest_cached_wizard_when_github_is_down(
    monkeypatch: pytest.MonkeyPatch, cache_home: Path, linux_arm64: None
) -> None:
    _cache(cache_home, "0.1.0")
    newest = _cache(cache_home, "0.10.0")
    runner = FakeRunner()

    def offline(_page: int) -> List[Any]:
        raise requests.ConnectionError("offline")

    github = FakeGitHub(b"", b"")
    _install(monkeypatch, github, runner)
    monkeypatch.setattr("deepeval.cli.setup.launcher._fetch_releases", offline)

    result = CliRunner().invoke(app, ["setup"])

    assert result.exit_code == 0, result.output
    assert "using the cached 0.10.0" in result.output
    assert runner.calls == [[str(newest)]]


@pytest.mark.parametrize(
    "pages", [[], [[_release("python-v4.3.0")]]], ids=["empty", "none"]
)
def test_no_release_and_no_cache_exits_1(
    monkeypatch: pytest.MonkeyPatch,
    cache_home: Path,
    linux_arm64: None,
    pages: List[List[Dict[str, Any]]],
) -> None:
    runner = FakeRunner()
    _install(monkeypatch, FakeGitHub(b"", b"", pages), runner)

    result = CliRunner().invoke(app, ["setup"])

    assert result.exit_code == 1
    assert "Could not find the setup wizard" in result.output
    assert runner.calls == []


def test_checksum_mismatch_writes_nothing(
    monkeypatch: pytest.MonkeyPatch, cache_home: Path, linux_arm64: None
) -> None:
    archive = _tar_gz({"deepeval-setup": BINARY})
    tampered = _tar_gz({"deepeval-setup": b"#!/bin/sh\necho evil\n"})
    runner = FakeRunner()
    _install(monkeypatch, FakeGitHub(tampered, _checksums(archive)), runner)

    result = CliRunner().invoke(app, ["setup"])

    assert result.exit_code == 1
    assert "Checksum mismatch" in result.output
    assert not cache_home.exists()
    assert runner.calls == []


@pytest.mark.parametrize(
    "members",
    [
        {"../deepeval-setup": BINARY},
        {"/deepeval-setup": BINARY},
        {"deepeval-setup": None},
        {"other": BINARY},
    ],
)
def test_rejects_unsafe_or_unexpected_archives(
    monkeypatch: pytest.MonkeyPatch,
    cache_home: Path,
    linux_arm64: None,
    members: Dict[str, Optional[bytes]],
) -> None:
    archive = _tar_gz(members)
    runner = FakeRunner()
    _install(monkeypatch, FakeGitHub(archive, _checksums(archive)), runner)

    result = CliRunner().invoke(app, ["setup"])

    assert result.exit_code == 1
    assert not _binary_path(cache_home).exists()
    assert runner.calls == []


@pytest.mark.parametrize(
    ("system", "machine"),
    [("Windows", "AMD64"), ("Linux", "ppc64le"), ("Darwin", "i386")],
)
def test_unsupported_platforms_exit_1(
    monkeypatch: pytest.MonkeyPatch,
    cache_home: Path,
    system: str,
    machine: str,
) -> None:
    monkeypatch.setattr(
        "deepeval.cli.setup.launcher.platform.system", lambda: system
    )
    monkeypatch.setattr(
        "deepeval.cli.setup.launcher.platform.machine", lambda: machine
    )
    github = FakeGitHub(b"", b"")
    runner = FakeRunner()
    _install(monkeypatch, github, runner)

    result = CliRunner().invoke(app, ["setup"])

    assert result.exit_code == 1
    assert "macOS and Linux" in result.output
    assert github.pages_requested == []
    assert github.requested == []
    assert runner.calls == []


@pytest.mark.parametrize(
    ("system", "machine", "target"),
    [
        ("Darwin", "x86_64", ("darwin", "x64")),
        ("Darwin", "arm64", ("darwin", "arm64")),
        ("Linux", "amd64", ("linux", "x64")),
        ("Linux", "aarch64", ("linux", "arm64")),
    ],
)
def test_maps_supported_platforms(
    system: str, machine: str, target: tuple
) -> None:
    assert launcher.resolve_target(system, machine) == target


def test_signal_exit_maps_to_shell_convention(
    monkeypatch: pytest.MonkeyPatch, cache_home: Path, linux_arm64: None
) -> None:
    archive = _tar_gz({"./deepeval-setup": BINARY})
    _install(
        monkeypatch,
        FakeGitHub(archive, _checksums(archive)),
        FakeRunner(returncode=-2),
    )

    assert CliRunner().invoke(app, ["setup"]).exit_code == 130


def test_cache_root_ignores_a_relative_xdg_cache_home(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    assert launcher.cache_root(env={"XDG_CACHE_HOME": "relative"}) == (
        Path.home() / ".cache" / "deepeval" / "setup"
    )
