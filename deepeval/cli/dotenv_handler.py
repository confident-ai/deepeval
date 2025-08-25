from __future__ import annotations
from pathlib import Path
import os
import re
import stat
from typing import Dict, Iterable

_LINE_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)\s*$")


class DotenvHandler:
    def __init__(self, path: str | Path = ".env.local"):
        self.path = Path(path)

    def _quote_if_needed(self, val: str) -> str:
        # keep existing quoting if present; add quotes for spaces/#
        if (val.startswith('"') and val.endswith('"')) or (
            val.startswith("'") and val.endswith("'")
        ):
            return val
        return f'"{val}"' if any(c in val for c in " #\t") else val

    def upsert(self, updates: Dict[str, str]) -> None:
        """
        Idempotently set/replace keys in a dotenv file. Preserves comments/order.
        Creates file if missing. Sets file mode to 0600 when possible.
        """
        lines = self.path.read_text().splitlines() if self.path.exists() else []
        seen = set()

        # replace existing keys in-place
        for i, line in enumerate(lines):
            m = _LINE_RE.match(line)
            if not m:
                continue
            key = m.group(1)
            if key in updates and key not in seen:
                lines[i] = f"{key}={self._quote_if_needed(updates[key])}"
                seen.add(key)

        # append missing keys at end (after a blank line)
        to_append = []
        for k, v in updates.items():
            if k in seen:
                continue
            to_append.append(f"{k}={self._quote_if_needed(v)}")
        if to_append:
            if lines and lines[-1].strip():
                lines.append("")
            lines.extend(to_append)

        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("\n".join(lines) + ("\n" if lines else ""))
        try:
            self.path.chmod(stat.S_IRUSR | stat.S_IWUSR)  # 0600
        except Exception:
            pass

    def unset(self, keys: Iterable[str]) -> None:
        """Remove keys from dotenv file, but leave comments and other lines untouched."""
        if not self.path.exists():
            return
        lines = self.path.read_text().splitlines()
        out = []
        keys = set(keys)
        for line in lines:
            m = _LINE_RE.match(line)
            if m and m.group(1) in keys:
                continue
            out.append(line)
        self.path.write_text("\n".join(out) + ("\n" if out else ""))
