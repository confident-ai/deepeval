import hashlib
import io
import json
import os
import stat
import subprocess
import tarfile
from pathlib import Path
from typing import Dict, List, Optional

import pytest
from typer.testing import CliRunner

from deepeval.cli.main import app
from deepeval.cli.setup import launcher

REPO_ROOT = Path(__file__).resolve().parents[3]
BINARY = b"#!/bin/sh\necho wizard\n"
ARCHIVE_NAME = "deepeval-setup-linux-arm64.tar.gz"


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


class FakeRelease:
    """Stands in for GitHub releases, so no test touches the network."""

    def __init__(self, archive: bytes, checksums: bytes) -> None:
        self.files = {
            launcher.release_url(ARCHIVE_NAME): archive,
            launcher.release_url(launcher.CHECKSUMS_NAME): checksums,
        }
        self.requested: List[str] = []

    def __call__(self, url: str) -> bytes:
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
    monkeypatch: pytest.MonkeyPatch, release: FakeRelease, runner: FakeRunner
) -> None:
    monkeypatch.setattr("deepeval.cli.setup.launcher._download", release)
    monkeypatch.setattr("deepeval.cli.setup.launcher.subprocess.run", runner)


def _binary_path(cache_home: Path) -> Path:
    return (
        cache_home
        / "deepeval"
        / "setup"
        / launcher.SETUP_WIZARD_VERSION
        / "deepeval-setup"
    )


def test_downloads_verifies_caches_and_runs_the_wizard(
    monkeypatch: pytest.MonkeyPatch, cache_home: Path, linux_arm64: None
) -> None:
    archive = _tar_gz({"deepeval-setup": BINARY})
    release = FakeRelease(archive, _checksums(archive))
    runner = FakeRunner(returncode=7)
    _install(monkeypatch, release, runner)

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
    assert release.requested == [
        "https://github.com/confident-ai/deepeval/releases/download/"
        f"setup-v{launcher.SETUP_WIZARD_VERSION}/{ARCHIVE_NAME}",
        "https://github.com/confident-ai/deepeval/releases/download/"
        f"setup-v{launcher.SETUP_WIZARD_VERSION}/SHA256SUMS",
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


def test_checksum_mismatch_writes_nothing(
    monkeypatch: pytest.MonkeyPatch, cache_home: Path, linux_arm64: None
) -> None:
    archive = _tar_gz({"deepeval-setup": BINARY})
    tampered = _tar_gz({"deepeval-setup": b"#!/bin/sh\necho evil\n"})
    runner = FakeRunner()
    _install(monkeypatch, FakeRelease(tampered, _checksums(archive)), runner)

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
    _install(monkeypatch, FakeRelease(archive, _checksums(archive)), runner)

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
    release = FakeRelease(b"", b"")
    runner = FakeRunner()
    _install(monkeypatch, release, runner)

    result = CliRunner().invoke(app, ["setup"])

    assert result.exit_code == 1
    assert "macOS and Linux" in result.output
    assert release.requested == []
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


def test_cached_binary_skips_the_download(
    monkeypatch: pytest.MonkeyPatch, cache_home: Path, linux_arm64: None
) -> None:
    binary = _binary_path(cache_home)
    binary.parent.mkdir(parents=True)
    binary.write_bytes(BINARY)
    binary.chmod(0o755)
    release = FakeRelease(b"", b"")
    runner = FakeRunner()
    _install(monkeypatch, release, runner)

    result = CliRunner().invoke(app, ["setup", "--org-id", "org"])

    assert result.exit_code == 0, result.output
    assert release.requested == []
    assert runner.calls == [[str(binary), "--org-id", "org"]]


def test_signal_exit_maps_to_shell_convention(
    monkeypatch: pytest.MonkeyPatch, cache_home: Path, linux_arm64: None
) -> None:
    archive = _tar_gz({"./deepeval-setup": BINARY})
    _install(
        monkeypatch,
        FakeRelease(archive, _checksums(archive)),
        FakeRunner(returncode=-2),
    )

    assert CliRunner().invoke(app, ["setup"]).exit_code == 130


def test_cache_dir_ignores_a_relative_xdg_cache_home(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    assert (
        launcher.cache_dir(env={"XDG_CACHE_HOME": "relative"})
        == (Path.home() / ".cache" / "deepeval" / "setup")
        / launcher.SETUP_WIZARD_VERSION
    )


def test_version_matches_the_setup_wizard_package() -> None:
    package = json.loads(
        (REPO_ROOT / "setup-wizard" / "package.json").read_text()
    )
    assert launcher.SETUP_WIZARD_VERSION == package["version"]
