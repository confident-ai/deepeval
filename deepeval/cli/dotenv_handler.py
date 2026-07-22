from __future__ import annotations

import os
import re
import stat
import tempfile
from io import StringIO
from pathlib import Path
from typing import Dict, Iterable, Mapping

from dotenv.parser import parse_stream

_KEY_LINE_RE = re.compile(
    r"^\s*(?:export\s+)?(?:'(?P<quoted>[^']+)'|(?P<key>[^=\#\s]+))\s*="
)
_BARE_VALUE_RE = re.compile(r"^[^\s#'\"\r\n]*$")
_ALLOWED_PLATFORM_SYMLINKS = {
    Path("/tmp"): Path("/private/tmp"),
    Path("/var"): Path("/private/var"),
}


class DotenvTargetError(ValueError):
    """Failure for unsafe dotenv save targets."""

    def __init__(self, path: Path):
        self.path = path
        super().__init__(
            f"Refusing to write to symlinked dotenv path: {path}. "
            "Point the configured save target, such as save= or "
            "DEEPEVAL_DEFAULT_SAVE, at the real dotenv file instead."
        )

    def cli_message(self) -> str:
        message = (
            f"Refusing to write to symlinked dotenv path: {self.path}. "
            "Point --save / DEEPEVAL_DEFAULT_SAVE at the real dotenv file "
            "instead."
        )
        return message


class DotenvHandler:
    def __init__(self, path: str | Path = ".env.local"):
        self.path = Path(path)

    def upsert(self, updates: Dict[str, str]) -> None:
        """
        Idempotently set/replace keys in a dotenv file. Preserves comments/order.
        Creates file if missing. Sets file mode to 0600 when possible.
        """
        self.update(updates=updates)

    def unset(self, keys: Iterable[str]) -> None:
        """Remove keys from dotenv file, but leave comments and other lines untouched."""
        self.update(removals=keys)

    def update(
        self,
        *,
        updates: Mapping[str, str] | None = None,
        removals: Iterable[str] = (),
    ) -> None:
        """
        Apply updates and removals in one atomic rewrite.

        This is used by settings/provider persistence paths where one logical
        operation can both write and clear keys.
        """
        updates = dict(updates or {})
        removals = set(removals)
        if not updates and not removals:
            return
        write_path = self.validate_target()
        if not updates and not write_path.exists():
            return
        self._rewrite(
            write_path=write_path,
            updates=updates,
            removals=removals,
            create=bool(updates),
        )

    def validate_target(self) -> Path:
        """Validate and return the path this handler is allowed to mutate."""
        return self._write_path()

    def _write_path(self) -> Path:
        """
        Return the path to mutate, refusing symlinked secret stores.

        Check the target file and its lexical parent chain so a configured save
        target cannot redirect persisted secrets through a symlinked path
        component. Known macOS system aliases are allowed because they are
        stable platform paths rather than project-controlled redirects.
        """
        for candidate in (self.path, *self.path.parents):
            if candidate.is_symlink() and not self._is_platform_alias(
                candidate
            ):
                raise DotenvTargetError(self.path)
        return self.path

    def _is_platform_alias(self, path: Path) -> bool:
        target = _ALLOWED_PLATFORM_SYMLINKS.get(path)
        if target is None:
            return False
        try:
            return path.resolve(strict=False) == target
        except OSError:
            return False

    def _rewrite(
        self,
        *,
        write_path: Path,
        updates: Mapping[str, str],
        removals: set[str],
        create: bool,
    ) -> None:
        if not create and not write_path.exists():
            return

        write_path.parent.mkdir(parents=True, exist_ok=True)
        original = (
            write_path.read_text(encoding="utf-8")
            if write_path.exists()
            else ""
        )
        content = self._render_updated_dotenv(original, updates, removals)
        self._atomic_write(write_path, content)

    def _render_updated_dotenv(
        self,
        original: str,
        updates: Mapping[str, str],
        removals: set[str],
    ) -> str:
        rendered: list[str] = []
        replaced: set[str] = set()

        for binding in parse_stream(StringIO(original)):
            original_line = binding.original.string
            key = binding.key or self._raw_assignment_key(original_line)
            if key is None:
                rendered.append(original_line)
                continue

            if binding.key is None and binding.error:
                _, separator, remainder = original_line.partition("\n")
                if separator and remainder:
                    if key in removals:
                        rendered.append(remainder)
                        continue
                    if key in updates:
                        if key not in replaced:
                            rendered.append(
                                self._format_line(key, updates[key])
                            )
                            replaced.add(key)
                        rendered.append(remainder)
                        continue

            if key in removals:
                continue
            if key in updates:
                if key not in replaced:
                    rendered.append(self._format_line(key, updates[key]))
                    replaced.add(key)
                continue
            rendered.append(original_line)

        for key, value in updates.items():
            if key in replaced:
                continue
            if rendered and not rendered[-1].endswith("\n"):
                rendered[-1] = f"{rendered[-1]}\n"
            rendered.append(self._format_line(key, value))

        return "".join(rendered)

    def _raw_assignment_key(self, line: str) -> str | None:
        match = _KEY_LINE_RE.match(line)
        if not match:
            return None
        return match.group("quoted") or match.group("key")

    def _format_line(self, key: str, value: str) -> str:
        value = self._escape_interpolation(value)
        # Keep shell-safe values bare for cross-tool compatibility, and quote
        # only when dotenv syntax requires escaping. Prefer single quotes
        # because they preserve backslash sequences such as JSON "\\n"; use
        # double quotes only when a quoted value ends with a backslash because
        # that would otherwise escape the closing single quote.
        if _BARE_VALUE_RE.fullmatch(value):
            value_out = value
        elif value.endswith("\\"):
            value_out = '"{}"'.format(
                value.replace("\\", "\\\\").replace('"', '\\"')
            )
        else:
            value_out = "'{}'".format(value.replace("'", "\\'"))
        return f"{key}={value_out}\n"

    def _escape_interpolation(self, value: str) -> str:
        # python-dotenv resolves ${VAR} after parsing regardless of quote style.
        # Encoding the literal "$" through an empty-name default preserves the
        # original value when DeepEval reads the file with interpolation enabled.
        return value.replace("${", "${:-$}{")

    def _atomic_write(
        self,
        path: Path,
        content: str,
    ) -> None:
        temp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                delete=False,
                prefix=".tmp_",
                dir=str(path.parent),
            ) as temp_file:
                temp_path = Path(temp_file.name)
                temp_file.write(content)
            temp_path.chmod(stat.S_IRUSR | stat.S_IWUSR)
            os.replace(temp_path, path)
            try:
                path.chmod(stat.S_IRUSR | stat.S_IWUSR)
            except OSError:
                # The replacement file was already hardened before os.replace.
                # Treat a post-replace chmod failure as best-effort so callers
                # do not roll back runtime state while the new dotenv content
                # has already been committed to disk.
                pass
        except BaseException:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)
            raise
